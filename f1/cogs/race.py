import sqlite3
from f1.api.stats import get_top_role_color
import time
from collections import defaultdict
from discord.ui import Button, View
import re
import requests
import wikipedia
from bs4 import BeautifulSoup
from unidecode import unidecode
import json
from fastf1.ergast import Ergast
import pytz
import fastf1
from f1.target import MessageTarget
from f1.errors import MissingDataError
from f1.config import Config
from f1.api import ergast, stats
from f1 import options, utils
from discord import default_permissions
import io
from discord.ext import commands
from discord import ApplicationContext, Embed
import pandas as pd
import discord
import matplotlib.pyplot as plt
import asyncio
import logging
from datetime import datetime
from datetime import date
import os
from dotenv import load_dotenv
import aiohttp
load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"
WIKI_REQUEST = 'http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='
current_date = date.today()
curr_year = int(datetime.now().year)
logger = logging.getLogger('f1-bot')

schedule = fastf1.get_event_schedule(
    int(datetime.now().year), include_testing=False)
last_index = None
for index, row in schedule.iterrows():
    if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
        last_index = index

SEASON = curr_year
ROUND = last_index


def parse_driver_name(name):
    return name.strip().replace("~", "").replace("*", "").replace("^", "")


def parse_championships(championships):
    champ_val = championships.split('<')[0].strip()[0]
    return int(champ_val) if champ_val.isdigit() else 0


def parse_brackets(text):
    return re.sub(r'\[.*?\]', '', text)


def get_wiki_image(search_term):
    try:
        result = wikipedia.search(search_term, results=1)
        wikipedia.set_lang('en')
        wkpage = wikipedia.WikipediaPage(title=result[0])
        title = wkpage.title
        response = requests.get(WIKI_REQUEST+title)
        json_data = json.loads(response.text)
        img_link = list(json_data['query']['pages'].values())[
            0]['original']['source']
        return img_link
    except:
        return 0


class ColDef:
    def __init__(self, name, width, textprops=None, border=None):
        self.name = name
        self.width = width
        self.textprops = textprops if textprops else {}
        self.border = border


def plot_table(dataframe, col_defs, index_col, figsize, background_color='black', text_color='white'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    fig.patch.set_facecolor(background_color)

    table_data = []
    column_widths = [col.width for col in col_defs]
    column_headers = [col.name for col in col_defs]

    # Create table data
    for _, row in dataframe.iterrows():
        table_data.append([row[col.name] for col in col_defs])

    # Plot the table
    table = ax.table(cellText=table_data, colLabels=column_headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Apply column definitions and styles
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if key[0] == 0:
            cell.set_facecolor(background_color)
            cell.set_text_props(weight='bold', color=text_color)
        else:
            cell.set_facecolor(background_color)
            cell.set_text_props(color=text_color)

        if key[1] < len(col_defs):
            col = col_defs[key[1]]
            cell.set_width(col.width)
            cell.set_text_props(**col.textprops)
            if col.border:
                if 'r' in col.border:
                    cell.set_linewidth(1)
                if 'l' in col.border:
                    cell.set_linewidth(1)

    return fig, ax


def get_driver(driver, ctx):
    try:
        # setup embed
        message_embed = discord.Embed(
            title="temp_driver_title", description="", color=get_top_role_color(ctx.author))

        url = 'https://en.wikipedia.org/wiki/List_of_Formula_One_drivers'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find_all('table')
        table = table[2]

        driver_data = []

        try:
            for row in table.find_all('tr')[1:]:
                columns = row.find_all('td')
                flags = row.find('img', {'class': 'mw-file-element'})
                if flags:
                    nationality = flags['src']
                if columns:
                    driver_dict = {
                        'name': parse_driver_name(columns[0].text.strip()),
                        'nationality': nationality,
                        'seasons_completed': columns[2].text.strip(),
                        'championships': parse_championships(columns[3].text.strip()),
                        'entries': parse_brackets(columns[4].text.strip()),
                        'starts': parse_brackets(columns[5].text.strip()),
                        'poles': parse_brackets(columns[6].text.strip()),
                        'wins': parse_brackets(columns[7].text.strip()),
                        'podiums': parse_brackets(columns[8].text.strip()),
                        'fastest_laps': parse_brackets(columns[9].text.strip()),
                        'points': parse_brackets(columns[10].text.strip())
                    }
                    driver_data.append(driver_dict)
        except Exception as e:
            message_embed.set_footer(f"Error getting data! {e}")

        normalized_input = unidecode(driver).casefold()
        # img_url = unidecode(driver).title()
        wiki_image = get_wiki_image(driver)

        # iterate through driver data to find a match
        index = -1
        for i in range(len(driver_data)):
            normalized_name = unidecode(driver_data[i]['name']).casefold()
            if normalized_name == normalized_input:
                index = i
                break
        if index == -1:

            message_embed.title = "Driver \"" + driver + "\" not found!"
            message_embed.description = "Try a driver's full name!"
            return message_embed
            # message_embed.timestamp = datetime.now()
            # message_embed.set_footer(text ='\u200b',icon_url="https://cdn.discordapp.com/attachments/884602392249770087/1059464532239581204/f1python128.png")
        else:
            if wiki_image != 0:
                message_embed.set_image(url=wiki_image)
                message_embed.set_thumbnail(
                    url=f"https:{driver_data[index]['nationality']}")
            message_embed.title = driver_data[index]['name']
            message_embed.description = ""
            message_embed.url = wikipedia.WikipediaPage(
                title=wikipedia.search(driver, results=1)[0]).url
            message_embed.add_field(
                name="**Seasons Completed:** ", value=str(driver_data[index]['seasons_completed']))
            message_embed.add_field(
                name="**Championships:** ", value=str(driver_data[index]['championships']))
            message_embed.add_field(
                name="**Entries:** ", value=str(driver_data[index]['entries']))
            message_embed.add_field(
                name="**Starts:** ", value=str(driver_data[index]['starts']))
            message_embed.add_field(
                name="**Poles:** ", value=str(driver_data[index]['poles']))
            message_embed.add_field(
                name="**Wins:** ", value=str(driver_data[index]['wins']))
            message_embed.add_field(
                name="**Podiums:** ", value=str(driver_data[index]['podiums']))
            message_embed.add_field(
                name="**Fastest Laps:** ", value=str(driver_data[index]['fastest_laps']))
            message_embed.add_field(
                name="**Points:**", value=str(driver_data[index]['points']))

            # message_embed.add_field(name = "Seasons Completed "+str(driver_data[index]['seasons_completed']),value=" ",inline = True)
            # message_embed.add_field(name = ":trophy: Championships "+str(driver_data[index]['championships']),value=" ",inline = True)
            # message_embed.add_field(name = ":checkered_flag: Entries "+str(driver_data[index]['entries']),value=" ",inline = True)
            # message_embed.add_field(name = ":checkered_flag: Starts "+str(driver_data[index]['starts']),value=" ",inline = True)
            # message_embed.add_field(name = ":stopwatch: Poles "+str(driver_data[index]['poles']),value=" ",inline = True)
            # message_embed.add_field(name = ":first_place: Wins "+str(driver_data[index]['wins']),value=" ",inline = True)
            # message_embed.add_field(name = ":medal: Podiums "+str(driver_data[index]['podiums']),value=" ",inline = True)
            # message_embed.add_field(name = ":purple_square: Fastest Laps "+str(driver_data[index]['fastest_laps']),value=" ",inline = True)
            # message_embed.add_field(name = ":chart_with_upwards_trend: Points "+str(driver_data[index]['points']),value=" ",inline = True)
            # message_embed.timestamp = datetime.now()
            # message_embed.set_footer(text ='\u200b',icon_url="https://cdn.discordapp.com/attachments/884602392249770087/1059464532239581204/f1python128.png")
            # print(message_embed)
            return message_embed
    except:
        print('Exception.')


stat_map = {
    'Starts': 'starts',
    'Career Points': 'careerpoints',
    'Wins': 'wins',
    'Podiums': 'podiums',
    'Poles': 'poles',
    'Fastest Laps': 'fastestlaps',
}


def get_drivers_standings():
    schedule = fastf1.get_event_schedule(
        int(datetime.now().year), include_testing=False)
    last_index = None
    for index, row in schedule.iterrows():
        if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
            last_index = index

    SEASON = curr_year
    ROUND = last_index
    ergast = Ergast()
    standings = ergast.get_driver_standings(season=SEASON, round=ROUND)
    return standings.content[0]


def calculate_max_points_for_remaining_season():
    curr_year = int(datetime.now().year)

    schedule = fastf1.get_event_schedule(
        int(datetime.now().year), include_testing=False)
    last_index = None
    for index, row in schedule.iterrows():
        if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
            last_index = index

    SEASON = curr_year
    ROUND = last_index
    POINTS_FOR_SPRINT = 8 + 25 + 1  # Winning the sprint, race and fastest lap
    POINTS_FOR_CONVENTIONAL = 25 + 1  # Winning the race and fastest lap
    events = fastf1.events.get_event_schedule(SEASON)
    events = events[events['RoundNumber'] > ROUND]

    # Count how many sprints and conventional races are left
    sprint_events = len(
        events.loc[events["EventFormat"] == "sprint_qualifying"])
    conventional_events = len(
        events.loc[events["EventFormat"] == "conventional"])

    # Calculate points for each
    sprint_points = sprint_events * POINTS_FOR_SPRINT
    conventional_points = conventional_events * POINTS_FOR_CONVENTIONAL

    return sprint_points + conventional_points


def calculate_who_can_win(driver_standings, max_points):
    LEADER_POINTS = int(driver_standings.loc[0]['points'])

    for i, _ in enumerate(driver_standings.iterrows()):
        driver = driver_standings.loc[i]
        driver_max_points = int(driver["points"]) + max_points
        can_win = 'No' if driver_max_points < LEADER_POINTS else 'Yes'

        driver_info = {
            "Position": driver["position"],
            "Driver": f"{driver['givenName']} {driver['familyName']}",
            "Current Points": driver["points"],
            "Theoretical max points": driver_max_points,
            "Can win?": can_win
        }

        # Yield the dictionary for the driver
        yield driver_info


connection = sqlite3.connect('guild_roles.db')
cursor = connection.cursor()

# Create table to map guilds to roles
cursor.execute('''
    CREATE TABLE IF NOT EXISTS GuildRoles (
        guild_id INTEGER NOT NULL,
        role_id INTEGER NOT NULL,
        PRIMARY KEY (guild_id, role_id)
    )
''')

# Create table to map guilds to a single channel
cursor.execute('''
    CREATE TABLE IF NOT EXISTS GuildChannels (
        guild_id INTEGER NOT NULL PRIMARY KEY,
        channel_id INTEGER NOT NULL
    )
''')

connection.commit()
connection.close()


def add_role_to_guild(guild_id, role_id):
    connection = sqlite3.connect('guild_roles.db')
    cursor = connection.cursor()

    # Insert role_id if it doesn't exist
    cursor.execute('''
        INSERT OR IGNORE INTO GuildRoles (guild_id, role_id)
        VALUES (?, ?)
    ''', (guild_id, role_id))

    connection.commit()
    connection.close()


def get_channel_and_roles_for_guild(guild_id):
    connection = sqlite3.connect('guild_roles.db')
    cursor = connection.cursor()

    cursor.execute('''
        SELECT channel_id FROM GuildChannels
        WHERE guild_id = ?
    ''', (guild_id,))

    channel_id = cursor.fetchone()
    if channel_id:
        channel_id = channel_id[0]
    else:
        channel_id = None

    cursor.execute('''
        SELECT role_id FROM GuildRoles
        WHERE guild_id = ?
    ''', (guild_id,))

    role_ids = [row[0] for row in cursor.fetchall()]
    connection.close()

    return channel_id, role_ids


def remove_role_from_guild(guild_id, role_id):
    connection = sqlite3.connect('guild_roles.db')
    cursor = connection.cursor()

    # Check if the role exists for the given guild_id
    cursor.execute('''
        SELECT 1 FROM GuildRoles
        WHERE guild_id = ? AND role_id = ?
    ''', (guild_id, role_id))

    exists = cursor.fetchone()

    if exists:
        # Remove the role_id for the given guild_id
        cursor.execute('''
            DELETE FROM GuildRoles
            WHERE guild_id = ? AND role_id = ?
        ''', (guild_id, role_id))

        connection.commit()
        connection.close()
        return True  # Indicate that the role was successfully removed
    else:
        connection.close()
        return False  # Indicate that the role was not found in the database


class Race(commands.Cog, guild_ids=Config().guilds):
    """All race related commands including qualifying, race results and pitstop data."""

    def __init__(self, bot: discord.Bot):
        self.bot = bot

    @commands.slash_command(description="Team Radio.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def radio(self, ctx: ApplicationContext, driver: options.DriverOptionRequired(), round: options.RoundOption, session: options.SessionOption, year: options.SeasonOption3):

        if round == None and year == None or round == None and year == int(datetime.now().year):

            schedule = fastf1.get_event_schedule(
                int(datetime.now().year), include_testing=False)

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    round = number
        if year == None:
            year = int(datetime.now().year)
        if year < int(datetime.now().year) and round == None:
            round = max(fastf1.get_event_schedule(
                int(year), include_testing=False)['RoundNumber'])
        await utils.check_season(ctx, year)
        if str(round).isdigit():
            round = int(round)
        ev = fastf1.get_event(year=year, gp=round)
        session1 = fastf1.get_session(year, ev['EventName'], session)
        session1.load(telemetry=False, laps=False, weather=False)
        if not driver.isdigit():

            driver_number = session1.get_driver(
                driver[0:3].upper())['DriverNumber']
        else:
            driver_number = driver

        url = f"https://api.openf1.org/v1/sessions?location={ev['Location']}&session_name={session}&year={year}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        a = data[0]['session_key']

        url = f"https://api.openf1.org/v1/team_radio?session_key={a}&driver_number={driver_number}"
        import json
        import base64

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    length = len(data)

                    options1 = []
                    for i in range(1, int(length)+1):
                        options1.append(discord.SelectOption(
                            label=str(i),
                            description=f"Pick radio number {i}"
                        ))

                    class MyView(discord.ui.View):
                        def __init__(self):
                            super().__init__()
                            self.radio_url = None

                        @discord.ui.select(  # the decorator that lets you specify the properties of the select menu
                            placeholder=0,  # the placeholder text that will be displayed if nothing is selected
                            min_values=1,  # the minimum number of values that must be selected by the users
                            max_values=1,  # the maximum number of values that can be selected by the users
                            options=options1
                        )
                        # the function called when the user is done selecting options
                        async def select_callback(self, select, interaction):
                            await interaction.response.send_message(f"You selected {select.values[0]}", ephemeral=True)

                            value = int(select.values[0])

                            mp3_url = data[value-1]['recording_url']
                            import subprocess

                            def create_video(mp3_bytes, image_bytes, output_file):

                                with open("temp.mp3", "wb") as mp3_temp, open("temp.png", "wb") as image_temp:
                                    mp3_temp.write(mp3_bytes)
                                    image_temp.write(image_bytes)
                                command = [
                                    'ffmpeg', '-y', '-loop', '1', '-i', 'temp.png', '-i', 'temp.mp3',
                                    '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', '-b:a', '192k',
                                    '-shortest', output_file
                                ]
                                subprocess.run(command, check=True)

                            async def send_video(ctx, mp3_url: str, image_url: str):
                                async with aiohttp.ClientSession() as session:
                                    # Download the MP3 file
                                    async with session.get(mp3_url) as response:
                                        mp3_bytes = await response.read()

                                    # Download the image file
                                    async with session.get(image_url) as response:
                                        image_bytes = await response.read()

                                output_file = "output.mp4"

                                # Create the video from the MP3 and image
                                create_video(
                                    mp3_bytes, image_bytes, output_file)

                                with open(output_file, "rb") as video_file:
                                    await msg.edit(file=discord.File(video_file, filename="output.mp4"))
                            await send_video(ctx, mp3_url, session1.get_driver(driver[0:3].upper())['HeadshotUrl'])

                embed = discord.Embed(title=f"Team Radio list", description=f"1 to {length} refers to the chronological order of radios for this driver in the selected session",
                                      color=get_top_role_color(ctx.author))
                msg = await ctx.respond(embed=embed, view=MyView())

    @commands.slash_command(description="Race Control data", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def racecontrol(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption,
                          session: options.SessionOption):

        if round == None and year == None or round == None and year == int(datetime.now().year):

            schedule = fastf1.get_event_schedule(
                int(datetime.now().year), include_testing=False)

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    round = number
        if year == None:
            year = int(datetime.now().year)
        if year < int(datetime.now().year) and round == None:
            round = max(fastf1.get_event_schedule(
                int(year), include_testing=False)['RoundNumber'])

        from discord.ext.pages import Paginator, Page
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        Session = await stats.load_session(ev, session, messages=True, laps=True, telemetry=True, weather=True)

        messages = Session.race_control_messages

        files = stats.racecontrol(messages, session)

        mypage = []
        for i in files:

            file = utils.plot_to_file(i, f"plot")
            embed = discord.Embed(
                title=f"{ev['EventDate'].year} {ev['EventName']} - {session} Race Control",
                color=get_top_role_color(ctx.author)
            )
            embed.set_image(url=f"attachment://plot.png")

            mypage.append(Page(embeds=[embed], files=[file]))

        paginator = Paginator(pages=mypage, timeout=None, author_check=False)
        try:
            await paginator.respond(ctx.interaction)
        except discord.Forbidden:
            return
        except discord.HTTPException:
            return

    @commands.slash_command(description="Tyre compound stints in a race.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def stints(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption,
                     driver: options.DriverOption):

        if type(round) == str and round.isdigit():
            round = int(round)
        if round == None and year == None or round == None and year == int(datetime.now().year):

            schedule = fastf1.get_event_schedule(
                int(datetime.now().year), include_testing=False)

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    round = number
        if year == None:
            year = int(datetime.now().year)
        if year < int(datetime.now().year) and round == None:
            round = max(fastf1.get_event_schedule(
                int(year), include_testing=False)['RoundNumber'])
        await utils.check_season(ctx, year)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, 'R', laps=True, telemetry=True)

        stints = await stats.tyre_stints(s, driver)

        if driver:
            stints.reset_index(inplace=True)
            table = stats.stints_driver(stints)
            f = utils.plot_to_file(table, "plot")
        else:
            # Group data as pivot table with laps driven per compound and indexed by driver
            # Does not show individual stints but total laps for each compound.
            pivot = pd.pivot_table(stints, values="TyreLife",
                                   index=["Driver"],
                                   columns="Compound",
                                   aggfunc="sum").fillna(0).astype(int)

            pivot.reset_index(inplace=True)
            table = stats.stints(pivot)
            f = utils.plot_to_file(table, "plot")
        embed = Embed(
            title=f"Race Tyre Stints - {event['EventName']} ({event['EventDate'].year})", color=get_top_role_color(ctx.author)
        )
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(name="wdc-contenders", description="Shows a list of drivers who can still mathematically win the wdc.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def whocanwinwdc(self, ctx: ApplicationContext):

        loop = asyncio.get_running_loop()
        driver_standings = await loop.run_in_executor(None, get_drivers_standings)

# Get the maximum amount of points
        points = await loop.run_in_executor(None, calculate_max_points_for_remaining_season)

        contenders = []

    # Iterate over the generator object and extract the relevant information
        for contender in await loop.run_in_executor(None, calculate_who_can_win, driver_standings, points):
            contenders.append(contender)
        a = stats.plot_chances(contenders)
        f = utils.plot_to_file(a, "plot")
        embed = discord.Embed(title='Theoretical WDC Contenders',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(description="Result data for the session. Default last race.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def results(self, ctx: ApplicationContext, year: options.SeasonOption2, round: options.RoundOption,
                      session: options.SessionOption):
        """Get the results for a session. The `round` can be the event name, location or round number in the season.
        The `session` is the identifier selected from the command choices.

        If no options given the latest race results will be returned, as defined by Ergast.

        Usage:
        ----------
            /results [year] [round] [session]
        """

        if type(round) == str and round.isdigit():
            round = int(round)
        if round == None and year == None or round == None and year == int(datetime.now().year):

            schedule = fastf1.get_event_schedule(
                int(datetime.now().year), include_testing=False)

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    round = number
        if year == None:
            year = int(datetime.now().year)
        if year < int(datetime.now().year) and round == None:
            round = max(fastf1.get_event_schedule(
                int(year), include_testing=False)['RoundNumber'])

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, session, laps=True, telemetry=True, messages=True, weather=True)

        data = await stats.format_results(s, session, year)

        table, ax = stats.results_table(data, session)
        ax.set_title(
            f"{ev['EventDate'].year} {ev['EventName']} - {session}"
        ).set_fontsize(13)

        f = utils.plot_to_file(table, "plot")
        embed = discord.Embed(
            title=f"{ev['EventDate'].year} {ev['EventName']} - {session}", color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(name="quizsetup")
    @default_permissions(administrator=True)
    async def quizsetup(self,
                        ctx: ApplicationContext,
                        role: discord.Option(discord.Role, "Select roles which can create quizzes ", required=False),
                        channel: discord.Option(discord.TextChannel, "Select a channel for the quizzes to go into", required=False),
                        remove_role: discord.Option(
                            discord.Role, "Select roles to remove", required=False)
                        ):
        if ctx.guild.name is not None:

            guild_id = ctx.guild.id
            role_id = role.id if role else None
            channel_id = channel.id if channel else None
            remove_role_id = remove_role.id if remove_role else None

            if role is not None:

                add_role_to_guild(guild_id, role_id)
            if remove_role_id:
                if remove_role_from_guild(guild_id, remove_role_id):
                    pass
                else:
                    await ctx.respond(f"Role {remove_role.mention} was not found in quiz permissions.", ephemeral=True)
                    return

            # Update channel if provided
            if channel_id is not None:
                connection = sqlite3.connect('guild_roles.db')
                cursor = connection.cursor()
                cursor.execute('''
                    INSERT INTO GuildChannels (guild_id, channel_id)
                    VALUES (?, ?)
                    ON CONFLICT(guild_id) DO UPDATE SET channel_id = ?
                ''', (guild_id, channel_id, channel_id))
                connection.commit()
                connection.close()

            await ctx.respond('Settings Updated for this server! Make sure you assign permissions to this bot for the specified channels if any.', ephemeral=True)
        else:
            await ctx.respond("Install this bot in your server to access quiz features!", ephemeral=True)

    @commands.slash_command(name="quiz")
    async def quiz(self, ctx: ApplicationContext, question: str, option1: str, option2: str, option3: str, option4: str, answer: options.quizoption, fact: discord.Option(str, required=False), image: discord.Option(discord.Attachment, "Upload an image", required=False) = None):
        if ctx.guild.name is not None:
            guild_id = ctx.guild.id
            channel_id, role_ids = get_channel_and_roles_for_guild(guild_id)

            if channel_id is None or role_ids is None:
                await ctx.respond("Quiz setup is not configured for this server.", ephemeral=True)
                return

            # Check if the user has any of the specific roles
            has_specific_role = any(
                role.id in role_ids for role in ctx.author.roles)

            if not has_specific_role:
                await ctx.respond("You don't have permission to create a quiz.", ephemeral=True)
                return

            # Fetch the channel to send the quiz
            channel = ctx.guild.get_channel(channel_id)
            await ctx.respond("Quiz Created!", ephemeral=True)

            class QuizButton(Button):
                def __init__(self, label, style, answer):
                    super().__init__(label=label, style=style)
                    self.answer = answer
                    count = {"1️⃣": 0, "2️⃣": 0, "3️⃣": 0, '4️⃣': 0}
                    self.count = count

                async def callback(self, interaction: discord.Interaction):
                    await interaction.response.send_message(f"You selected {self.label}", ephemeral=True)
                    self.view.selected_answer = self.label
                    self.view.user_responses[interaction.user] = self.label
                    self.count[self.label] += 1

            class QuizView(View):

                def __init__(self, answer):
                    super().__init__()
                    self.answer = answer
                    self.user_responses = defaultdict(int)

                async def on_timeout(self):
                    # Create a new view for displaying results
                    result_view = View()
                    option_counts = defaultdict(int)

                    # Count the responses
                    for option in self.user_responses.values():
                        option_counts[option] += 1

                    # Update button styles and labels
                    for item in self.children:
                        if isinstance(item, QuizButton):

                            style = discord.ButtonStyle.success if item.label == self.answer else discord.ButtonStyle.danger
                            label = f"{item.label} ({item.count.get(item.label)})"
                            result_view.add_item(QuizButton(
                                label=label, style=style, answer=item.answer))

                    # Edit the message to remove the countdown field
                    for item in result_view.children:
                        if isinstance(item, QuizButton):
                            item.disabled = True

                    # Edit the message to remove the countdown field
                    if self.message:
                        embed = self.message.embeds[0]

                        for index, field in enumerate(embed.fields):
                            if "Time Remaining" in field.name:
                                embed.remove_field(index)
                                break
                        if embed.image:

                            img_data1 = await image.read()
                            img1 = discord.File(io.BytesIO(
                                img_data1), filename="quiz_image.png")
                            embed.set_image(url="attachment://quiz_image.png")
                            await self.message.edit(embed=embed, file=img1, view=result_view)
                        else:
                            await self.message.edit(embed=embed, view=result_view)
            # Send quiz embed with custom question and options
            embed = discord.Embed(title="You only have 10 seconds to answer!",
                                  description=f"**{question}**", color=discord.Color.blurple())
            embed.set_author(name='FormulaOne Quiz')
            embed.add_field(name="1️⃣", value=option1, inline=False)
            embed.add_field(name="2️⃣", value=option2, inline=False)
            embed.add_field(name="3️⃣", value=option3, inline=False)
            embed.add_field(name="4️⃣", value=option4, inline=False)

        # Create the embed with the countdown timestamp

            embed.set_footer(
                text='Only first 5 users to answer correctly are mentioned!')
            # Set timestamp for live countdown

            embed2 = discord.Embed(title="Quiz starting in 10 seconds!",
                                   description="", color=discord.Color.blurple())
            await channel.send(embed=embed2, delete_after=10)
            future_time = int(time.time()) + 20
            embed.add_field(name="Time Remaining",
                            value=f"<t:{future_time}:R>", inline=False)
            await asyncio.sleep(10)

            if image:
                img_data = await image.read()
                img = discord.File(io.BytesIO(img_data), filename="quiz_image.png")
                embed.set_image(url="attachment://quiz_image.png")

            view = QuizView(answer)
            buttons = [
                QuizButton(
                    label="1️⃣", style=discord.ButtonStyle.primary, answer=option1),
                QuizButton(
                    label="2️⃣", style=discord.ButtonStyle.primary, answer=option2),
                QuizButton(
                    label="3️⃣", style=discord.ButtonStyle.primary, answer=option3),
                QuizButton(
                    label="4️⃣", style=discord.ButtonStyle.primary, answer=option4)
            ]
            for button in buttons:
                view.add_item(button)

            if image:
                quiz_message = await channel.send(embed=embed, file=img, view=view)
            else:
                quiz_message = await channel.send(embed=embed, view=view)

            await asyncio.sleep(10)
            await view.on_timeout()

            correct_answers = 0
            winners = []
            for user, user_answer in view.user_responses.items():
                if user_answer == answer:
                    winners.append(user)
                    correct_answers += 1

            winner_embed = discord.Embed(
                title="Quiz Results", color=discord.Color.blurple())
            if not fact:
                fact = ""
            d = {"1️⃣": option1, "2️⃣": option2, "3️⃣": option3, "4️⃣": option4}
            if correct_answers == 0:
                winner_embed.description = f"No one answered correctly. The correct answer was **||{d.get(answer)}. {fact}||**"
            else:
                winner_embed.description = f"The correct answer was **||{d.get(answer)}. {fact}||**. {correct_answers} {'person' if correct_answers == 1 else 'people'} answered correctly:"
                for i, winner in enumerate(winners):
                    if i < 5:  # Mention up to 5 winners
                        winner_embed.description += f"\n{i + 1}. {winner.mention}"

            await quiz_message.reply(embed=winner_embed)
        else:
            await ctx.respond("Install this bot in your server to access quiz features!", ephemeral=True)

    @commands.slash_command(name="reddit", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def meme2(self, ctx: ApplicationContext):
        import asyncpraw
        import random

        async def get_random_post():
            reddit = asyncpraw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
            subreddit = await reddit.subreddit('formuladank')
            posts = [post async for post in subreddit.hot(limit=50)]
            random_post = random.choice(posts)

            while random_post.is_self or not random_post.url.lower().endswith(('jpg', 'jpeg', 'png', 'gif')):
                # Ensure the post is an image or GIF
                random_post = random.choice(posts)

            return random_post

        post = await get_random_post()
        embed = discord.Embed(title=post.title, url=post.url,
                              color=get_top_role_color(ctx.author))
        embed.set_image(url=post.url)
        embed.set_footer(
            text=f"👍 {post.score} | 💬 {post.num_comments} | Posted by u/{post.author}")
        await ctx.respond(embed=embed)

    @commands.slash_command(description="Race pitstops ranked by duration or filtered to a driver.", name="pitstops", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def pitstops(self, ctx: ApplicationContext, year: options.SeasonOption, round: options.RoundOption,
                       filter: options.RankedPitstopFilter, driver: options.DriverOption):
        """Display pitstops for the race ranked by `filter` or `driver`.

        All parameters are optional. Defaults to the best pitstop per driver for the most recent race.
        Pitstop data unavailable before 2012.

        Usage:
        ----------
            /pitstops-ranked [season] [round] [filter]
        """

        if round == None and year == None or round == None and year == int(datetime.now().year):

            schedule = fastf1.get_event_schedule(
                int(datetime.now().year), include_testing=False)

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    round = number
        if year == None:
            year = int(datetime.now().year)
        if year < int(datetime.now().year) and round == None:
            round = max(fastf1.get_event_schedule(
                int(year), include_testing=False)['RoundNumber'])
        if not year == 'current':
            if int(year) < 2012:
                raise commands.BadArgument(
                    message="Pitstop data unavailable before 2012.")
        await utils.check_season(ctx, year)

        # Get event info to match race name idenfifiers from command
        event = await stats.to_event(year, round)
        yr, rd = event["EventDate"].year, event["RoundNumber"]

        # Process pitstop data
        data = await stats.filter_pitstops(yr, rd, filter, driver)


# Join the formatted pitstop times into a single string

        table, ax = stats.pitstops_table(data)
        ax.set_title(
            f"{yr} {event['EventName']} | Pitstops ({filter})"
        ).set_fontsize(13)

        f = utils.plot_to_file(table, "plot")
        embed = discord.Embed(
            title=f"{yr} {event['EventName']} | Pitstops", color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(description="Best ranked lap times per driver.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def laptimes(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption,
                       tyre: options.TyreOption, session: options.SessionOption):
        """Best ranked lap times per driver in the race. All parameters optional.

        Only the best recorded lap for each driver in the race.

        Usage:
        ----------
            /laptimes [season] [round] [tyre]
        """
        if round == None and year == None or round == None and year == int(datetime.now().year):

            schedule = fastf1.get_event_schedule(
                int(datetime.now().year), include_testing=False)

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    round = number
        if year == None:
            year = int(datetime.now().year)
        if year < int(datetime.now().year) and round == None:
            round = max(fastf1.get_event_schedule(
                int(year), include_testing=False)['RoundNumber'])
        await utils.check_season(ctx, year)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True)
        data = stats.fastest_laps(s, tyre)

        # Get the table
        table, ax = stats.laptime_table(data)
        ax.set_title(
            f"{event['EventDate'].year} {event['EventName']}\nFastest Lap Times"
        ).set_fontsize(13)

        f = utils.plot_to_file(table, "plot")
        embed = discord.Embed(
            title=f"{event['EventDate'].year} {event['EventName']} Fastest Lap Times", color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(
        description="View fastest sectors and speed trap based on quick laps. Seasons >= 2018.", integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        })
    async def sectors(self, ctx: ApplicationContext, year: options.SeasonOption3,
                      round: options.RoundOption, tyre: options.TyreOption, session: options.SessionOption):
        """View min sector times and max speedtrap per driver. Based on recorded quicklaps only."""
        if round == None and year == None or round == None and year == int(datetime.now().year):

            schedule = fastf1.get_event_schedule(
                int(datetime.now().year), include_testing=False)

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    round = number
        if year == None:
            year = int(datetime.now().year)
        if year < int(datetime.now().year) and round == None:
            round = max(fastf1.get_event_schedule(
                int(year), include_testing=False)['RoundNumber'])
        ev = await stats.to_event(year, round)
        yr, rd = ev["EventDate"].year, ev["RoundNumber"]
        s = await stats.load_session(ev, session, laps=True)
        data = stats.sectors(s, tyre)

        table, ax = stats.sectors_table(data)
        ax.set_title(
            f"{yr} {ev['EventName']} - Sectors" +
            (f"\nTyre: {tyre}" if tyre else "")
        ).set_fontsize(12)

        f = utils.plot_to_file(table, "plot")
        embed = discord.Embed(title=f'Sectors and Speed Trap: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(description="Career stats for a driver. Enter Full Name of the driver.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def career(self, ctx: ApplicationContext, driver: options.driveroption2):

        loop = asyncio.get_running_loop()
        result_embed = await loop.run_in_executor(None, get_driver, driver, ctx)
        # send final embed
        await ctx.respond(embed=result_embed)

    @commands.slash_command(
        name="track-incidents",
        description="Summary of race events including Safety Cars and retirements.", integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        })
    async def track_incidents(self, ctx: ApplicationContext,
                              year: options.SeasonOption3, round: options.RoundOption):
        """Outputs a table showing the lap number and event, such as Safety Car or Red Flag."""
        if round == None and year == None or round == None and year == int(datetime.now().year):

            schedule = fastf1.get_event_schedule(
                int(datetime.now().year), include_testing=False)

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    round = number
        if year == None:
            year = int(datetime.now().year)
        if year < int(datetime.now().year) and round == None:
            round = max(fastf1.get_event_schedule(
                int(year), include_testing=False)['RoundNumber'])
        await utils.check_season(ctx, year)
        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R", laps=True, messages=True)
        # Assuming 'incidents' is previously defined as shown in your snippet

        if not s.f1_api_support:
            raise MissingDataError("Track status data unavailable.")

        # Get the driver DNFs
        dnfs = stats.get_dnf_results(s)
        # Combine driver code and dnf reason into single column for merging
        dnfs["Event"] = dnfs.apply(
            lambda row: f"Retired: {row['Abbreviation']} ({row['Status']})", axis=1)
        dnfs = dnfs.loc[:, ["LapNumber", "Event"]]

        # Get track status events grouped by lap number
        track_events = stats.get_track_events(s)

        if dnfs["Event"].size == 0 and track_events["Event"].size == 0:
            raise MissingDataError("No track events in this race.")

        # Combine the driver retirements and track status events
        incidents = pd.concat([
            dnfs.loc[:, ["LapNumber", "Event"]],
            track_events.loc[:, ["LapNumber", "Event"]]
        ], ignore_index=True).sort_values(by="LapNumber").reset_index(drop=True)
        incidents["LapNumber"] = incidents["LapNumber"]
        incidents["LapNumber"] = incidents["LapNumber"].fillna("No data")
        table, ax = stats.incidents_table(incidents)
        ax.set_title(
            f"{ev['EventDate'].year} {ev['EventName']}\nTrack Incidents"
        ).set_fontsize(12)

        f = utils.plot_to_file(table, "plot")
        embed = discord.Embed(
            title=f"{ev['EventDate'].year} {ev['EventName']} Track Incidents", color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        embed.set_footer(
            text=r"If you see erroneous data or want a detailed analysis of session events, use /racecontrol. No data represents the driver did not start the race.")
        await ctx.respond(embed=embed, file=f)


def setup(bot: discord.Bot):
    bot.add_cog(Race(bot))
