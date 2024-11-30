import urllib.parse
from datetime import datetime, date
import pytz
from f1.api.stats import get_top_role_color
from f1.utils import F1_RED, check_season
from f1.target import MessageTarget
from f1.config import Config
from f1.api import ergast, stats
from f1 import utils
from f1 import options
from discord import Embed
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
import os
import matplotlib.pyplot as plt
import logging
import discord
import fastf1
from fastf1.ergast import Ergast
import pandas as pd
from datetime import date
from datetime import datetime
import country_converter as coco
import requests
import json
from plottable import ColDef, Table
import config
from discord.ext import commands
import asyncio
from tabulate import tabulate
import requests
import discord
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import typing
from discord.ext import commands
import fitz
import uuid
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"


now = pd.Timestamp.now()


fastf1.Cache.enable_cache('cache/')
logger = logging.getLogger("f1-bot")


def schedule(ctx):
    now = pd.Timestamp.now()

    message_embed = discord.Embed(
        title="Schedule", description="", color=get_top_role_color(ctx.author))

    message_embed.set_author(name='F1 Race Schedule')

    schedule = fastf1.get_event_schedule(
        int(datetime.now().year), include_testing=False)

    # index of next event (round number)
    next_event = 0

    for index, row in schedule.iterrows():
        current_date = date.today()
        import pytz
        if row["Session5Date"] < pd.Timestamp(current_date, tzinfo=pytz.utc):
            number = row['RoundNumber']
            next_event = number+1

    try:

        if (len(schedule) <= next_event):
            raise IndexError

        # else
        # get name of event
        race_name = schedule.loc[next_event, "EventName"]

        # get emoji for country
        emoji = ":flag_" + \
            (coco.convert(
                names=schedule.loc[next_event, "Country"], to='ISO2')).lower()+":"

        # Rename embed title
        message_embed.title = "Race Schedule for " + race_name

        # create a dictionary to store converted times
        # adjust emojis/session name according to weekend format
        if (schedule.loc[next_event, "EventFormat"] == 'conventional'):
            converted_session_times = {
                f":one: {schedule.loc[next_event, 'Session1']}": schedule.loc[next_event, "Session1Date"],
                f":two: {schedule.loc[next_event, 'Session2']}": schedule.loc[next_event, "Session2Date"],
                f":three: {schedule.loc[next_event, 'Session3']}": schedule.loc[next_event, "Session3Date"],
                f":stopwatch: {schedule.loc[next_event, 'Session4']}": schedule.loc[next_event, "Session4Date"],
                f":checkered_flag: {schedule.loc[next_event, 'Session5']}": schedule.loc[next_event, "Session5Date"]
            }
            # fp1_date = pd.Timestamp(converted_session_times[":one: FP1"]).strftime('%Y-%m-%d')
            # fp2_date = pd.Timestamp(converted_session_times[":two: FP2"]).strftime('%Y-%m-%d')
            # fp3_date = pd.Timestamp(converted_session_times[":three: FP3"]).strftime('%Y-%m-%d')
            # quali_date = pd.Timestamp(converted_session_times[":stopwatch: Qualifying"]).strftime('%Y-%m-%d')
            # race_date = pd.Timestamp(converted_session_times[":checkered_flag: Race"]).strftime('%Y-%m-%d')
        else:
            converted_session_times = {
                f":one: {schedule.loc[next_event, 'Session1']}": schedule.loc[next_event, "Session1Date"],
                f":stopwatch: {schedule.loc[next_event, 'Session2']}": schedule.loc[next_event, "Session2Date"],
                f":stopwatch: {schedule.loc[next_event, 'Session3']}": schedule.loc[next_event, "Session3Date"],
                f":race_car: {schedule.loc[next_event, 'Session4']}": schedule.loc[next_event, "Session4Date"],
                f":checkered_flag: {schedule.loc[next_event, 'Session5']}": schedule.loc[next_event, "Session5Date"]
            }
            # fp1_date = pd.Timestamp(converted_session_times[":one: FP1"]).strftime('%Y-%m-%d')
            # fp2_date = pd.Timestamp(converted_session_times[":two: FP2"]).strftime('%Y-%m-%d')
            # quali_date = pd.Timestamp(converted_session_times[":stopwatch: Qualifying"]).strftime('%Y-%m-%d')
            # sprint_date = pd.Timestamp(converted_session_times[":race_car: Sprint"]).strftime('%Y-%m-%d')
            # race_date = pd.Timestamp(converted_session_times[":checkered_flag: Race"]).strftime('%Y-%m-%d')

        # string to hold description message
        time_until = schedule.loc[next_event, "Session5Date"].tz_convert(
            'America/New_York') - now.tz_localize('America/New_York')
        out_string = countdown(time_until.total_seconds())

        try:
            location = schedule.loc[next_event, "Location"]
            # try to get timezone from list
            # local_tz = timezones.timezones_list[location]
            # print("Getting timezone from timezones.py")
            # convert times to EST
            for key in converted_session_times.keys():
                date_object = converted_session_times.get(
                    key).tz_convert('America/New_York')
                converted_session_times.update({key: date_object})

        # timezone not found in FastF1 <-- should not be possible anymore
        except Exception as e:
            print("Using fallback timezone calculation")
            # get location of race
            print(e)
            # calculate timezone using latitude/longitude
            convert_timezone_fallback(location, converted_session_times)

        # strings to store session names and times
        sessions_string = ''
        times_string = ''

        # setup strings to be added to fields
        # strings to store session names and times''

# setup strings to be added to fields
        for key in converted_session_times.keys():
            # Convert timestamp to datetime object
            timestamp = converted_session_times.get(key).timestamp()
            abc = int(timestamp)
            times_string += f"<t:{abc}:R> <t:{abc}:F>\n"
            sessions_string += key + '\n'

        message_embed.add_field(
            name="Session", value=sessions_string, inline=True)
        message_embed.add_field(
            name="Time", value=times_string, inline=True)
        message_embed.add_field(
            name="Track Layout", value="", inline=False)
        country = schedule.loc[next_event, "Country"]
        message_embed.set_image(url=get_circuit_image(
            location, country).replace(" ", "%20"))

        # add fields to embed

    except:
        return out_string

    return message_embed


def get_circuit_image(location, country):
    if country.lower() == 'united kingdom':
        country = 'Great Britain'
    current_year = datetime.now().year
    urls = [
        f"https://www.formula1.com/en/racing/{current_year}/{location.replace(' ', '-')}/circuit.html",
        f"https://www.formula1.com/en/racing/{current_year}/{country.replace(' ', '-')}/circuit.html"
    ]

    session = requests.Session()
    for url in urls:
        try:
            response = session.get(url, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            img_tag = next(
                (img for img in soup.find_all('img')
                 if "circuit" in img.get('src').lower()), None
            )
            if img_tag:
                return urllib.parse.urljoin(url, img_tag['src'])
        except Exception as e:
            print(f"Error fetching image from {url}: {e}")
    return None


def json2obj(json_file):
    return json.loads(json.dumps(json_file))


curr_year = int(datetime.now().year)


message_embed = discord.Embed(title=f"FIA Document(s)", description="")
message_embed.set_author(name='FIA docs')


def get_fia_doc(doc=None):
    from io import BytesIO
    message_embed.title = f"FIA Document {doc}"

    if doc is None:
        doc = 0
        message_embed.title = "Latest FIA Document"

    # get FIA site
    url = 'https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043'
    html = requests.get(url=url)
    s = BeautifulSoup(html.content, 'html.parser')

    # get latest document
    results = s.find_all(class_='document-row')
    documents = [result.find('a')['href']
                 for result in results if result.find('a')]

    if doc >= len(documents):
        raise IndexError("Document index out of range")

    doc_url = 'https://www.fia.com/' + documents[doc]
    print(doc_url)

    # create a file name for it
    fileName = documents[doc].split(
        '/sites/default/files/decision-document/')[-1]
    message_embed.description = fileName[:-4]

    images = []

    # fetch the document
    doc_response = requests.get(doc_url)
    pdf_stream = io.BytesIO(doc_response.content)

    # convert pdf to images
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    page_num = 0
    try:
        while True:
            page = doc.load_page(page_num)  # number of page
            pix = page.get_pixmap(matrix=(fitz.Matrix(300 / 72, 300 / 72)))
            img_stream = io.BytesIO()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(img_stream, format="PNG")
            img_stream.seek(0)  # Go to the beginning of the BytesIO stream
            images.append(discord.File(img_stream, filename=f"{page_num}.png"))
            page_num += 1
    except ValueError:
        pass
    doc.close()

    return images


def convert_timezone_fallback(location, converted_session_times):
    # create coordinate finder object
    g = Nominatim(user_agent='f1pythonbottesting')
    # get coordinates of track
    coords = g.geocode(location)
    print(location + ": " + (str)(coords.latitude) +
          " , " + (str)(coords.longitude))

    # create timezone finder object
    tf = TimezoneFinder()

    # find track timezone using its coordinates
    tz = tf.timezone_at(lng=coords.longitude, lat=coords.latitude)
    # update dictionary with converted times
    for key in converted_session_times.keys():
        date_object = converted_session_times.get(key).tz_localize(
            tz).tz_convert('America/New_York')
        converted_session_times.update({key: date_object})
        # print(date_object)

# function that takes a timedelta and returns a countdown string


def countdown(totalseconds):
    out_string = ""
    days = int(totalseconds // 86400)
    totalseconds %= 86400
    hours = int(totalseconds // 3600)
    totalseconds %= 3600
    minutes = int(totalseconds // 60)
    seconds = totalseconds % 60
    seconds = int(seconds // 1)
    out_string += f"{days} days, {hours} hours, {minutes} minutes, and {seconds} seconds until the race!"
    return out_string


class Season(commands.Cog, guild_ids=Config().guilds):
    """Commands related to F1 season, e.g. championship standings and schedule."""

    def __init__(self, bot: discord.Bot):
        self.bot = bot

    @commands.slash_command(description="Driver championship standings.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def wdc(self, ctx, year: options.SeasonOption):
        """Display the Driver Championship standings as of the last race or `season`.


        Usage:
        ------
            /wdc [season]    WDC standings from [season].
        """
        if year == None:
            year = int(datetime.now().year)
        await check_season(ctx, year)

        schedule = fastf1.get_event_schedule(year, include_testing=False)

        if year == int(datetime.now().year):

            import pytz

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    last_index = number

            last_round = last_index
        else:
            last_round = max(schedule.RoundNumber)

        result = await ergast.get_driver_standings(year)
        table, ax = stats.championship_table(result['data'], type="wdc")
        yr, rd = result['season'], result['round']
        ax.set_title(f"{yr} Driver Championship - Round {rd}").set_fontsize(12)

        f = utils.plot_to_file(table, "plot")
        embed = discord.Embed(
            title=f'WDC standings for the {yr} season', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(file=f, embed=embed)

    @commands.slash_command(description="Provides season calender", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def calender(self, ctx, year: options.SeasonOption2):
        if year == None:
            year = int(datetime.now().year)
        await check_season(ctx, year)

        calender = await ergast.get_race_schedule2(year)
        table, ax = stats.plot_race_schedule(calender['data'])
        ax.set_title(f"{year} Calender").set_fontsize(12)
        f = utils.plot_to_file(table, "plot")

        embed = discord.Embed(
            title=f'Calender {year}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")

        await ctx.respond(file=f, embed=embed)

    @commands.slash_command(name='fiadoc', description='Get latest FIA Document. Use 1, 2, 3.. for older documents.', integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def fiadoc(self, ctx, doc: typing.Optional[int]):

        from discord.ext.pages import Paginator, Page
        mypage = []
        loop = asyncio.get_running_loop()
        images = await loop.run_in_executor(None, get_fia_doc, doc)
        a = 0
        for i in images:

            embed = discord.Embed(title=f"FIA Document(s)", description="", color=get_top_role_color(ctx.author)).set_thumbnail(
                url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg/1200px-F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg.png')
            embed.set_image(url=f"attachment://{a}.png")

            mypage.append(Page(embeds=[embed], files=[i]))
            a += 1

        paginator = Paginator(pages=mypage, timeout=None, author_check=False)
        try:
            await paginator.respond(ctx.interaction)
        except discord.Forbidden:
            return
        except discord.HTTPException:
            return

    @commands.slash_command(description="Constructors Championship standings.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def wcc(self, ctx, year: options.SeasonOption):
        """Display Constructor Championship standings as of the last race or `season`.

        Usage:
        ------
            /wcc [season]   WCC standings from [season].
        """

        if year == None:
            year = int(datetime.now().year)
        await check_season(ctx, year)

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if year == int(datetime.now().year):

            import pytz

            for index, row in schedule.iterrows():

                if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                    number = row['RoundNumber']
                    last_index = number

            last_round = last_index
        else:
            last_round = max(schedule.RoundNumber)
        result = await ergast.get_team_standings(year, last_round)
        table, ax = stats.championship_table(result['data'], type="wcc")
        yr, rd = result['season'], result['round']
        ax.set_title(
            f"{yr} Constructor Championship - Round {rd}").set_fontsize(12)

        f = utils.plot_to_file(table, "plot")
        embed = discord.Embed(
            title=f'WCC standings for the {yr} season', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(file=f, embed=embed)

    @commands.slash_command(description='get race schedule', integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def schedule(self, ctx):
        loop = asyncio.get_running_loop()
        message_embed = await loop.run_in_executor(None, schedule, ctx)
        await ctx.respond(embed=message_embed)


def setup(bot: discord.Bot):
    bot.add_cog(Season(bot))
