import sqlite3
from f1.api.stats import get_top_role_color
import time
from collections import defaultdict
from discord.ui import Button, View
import fastf1
from f1.errors import MissingDataError
from f1.api.stats import get_channel_and_roles_for_guild, get_drivers_standings, calculate_max_points_for_remaining_season, calculate_who_can_win, get_driver
from f1.config import Config
from f1.api import ergast, stats
from f1.api.stats import get_ephemeral_setting
from f1 import options, utils
from f1.api.stats import add_role_to_guild, remove_role_from_guild
from discord import default_permissions
import io
from f1.api.stats import roundnumber
from discord.ext import commands
from discord import ApplicationContext, Embed
import pandas as pd
import discord
import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import aiohttp
load_dotenv()
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"
logger = logging.getLogger('f1-bot')


class Race(commands.Cog, guild_ids=Config().guilds):
    """All race related commands including qualifying, race results and pitstop data."""

    def __init__(self, bot: discord.Bot):
        self.bot = bot

    @commands.slash_command(description="Team Radio.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def radio(self, ctx: ApplicationContext, driver: options.DriverOptionRequired(), round: options.RoundOption, session: options.SessionOption, year: options.SeasonOption5):

        from discord.ext.pages import Paginator, Page
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        if str(round).isdigit():
            round = int(round)
        ev = fastf1.get_event(year=year, gp=round)
        s = fastf1.get_session(year, ev['EventName'], session)
        s.load(telemetry=False, laps=False, weather=False)
        if not driver.isdigit():

            driver_number = s.get_driver(
                driver[0:3].upper())['DriverNumber']
            driver_name = s.get_driver(
                driver[0:3].upper())['LastName']
        else:
            driver_number = driver
            driver_name = s.get_driver(
                driver)['LastName']

        url = f"https://api.openf1.org/v1/sessions?location={ev['Location']}&session_name={session}&year={year}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        a = data[0]['session_key']

        url = f"https://api.openf1.org/v1/team_radio?session_key={a}&driver_number={driver_number}"
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

        async def send_video(mp3_url: str, image_url: str):
            async with aiohttp.ClientSession() as session:
                # Download the MP3 file
                async with session.get(mp3_url) as response:
                    mp3_bytes = await response.read()

                # Download the image file
                async with session.get(image_url) as response:
                    image_bytes = await response.read()

            output_file = "output.mp4"
            create_video(
                mp3_bytes, image_bytes, output_file)

            with open(output_file, "rb") as video_file:
                return discord.File(video_file, filename="output.mp4")

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    length = len(data)
                    mypage = []
                    for i in range(0, length):
                        embed = discord.Embed(
                            title=f"{ev['EventDate'].year} {ev['EventName']} Radios for {driver_name.title()}",
                            color=get_top_role_color(ctx.author)
                        )
                        mp3_url = data[i]['recording_url']
                        file = await send_video(mp3_url, s.get_driver(driver[0:3].upper())['HeadshotUrl'])

                        mypage.append(Page(embeds=[embed], files=[file]))

                    paginator = Paginator(
                        pages=mypage, timeout=None, author_check=False)
                    try:
                        await paginator.respond(ctx.interaction)
                    except discord.Forbidden:
                        return
                    except discord.HTTPException:
                        return

    @commands.slash_command(description="Race Control data", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def racecontrol(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption,
                          session: options.SessionOption):

        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]

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
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="wdc-contenders", description="Shows a list of drivers who can still mathematically win the wdc.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def whocanwinwdc(self, ctx: ApplicationContext):
        try:

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
            await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))
        except:
            await ctx.respond("Season has finished or is yet to start!", ephemeral=get_ephemeral_setting(ctx))

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

        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        ev = await stats.to_event(year, round)
        if session in ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Sprint", "Race"]:
            s = await stats.load_session(ev, session, laps=False, telemetry=False, messages=False, weather=False)
        else:
            s = await stats.load_session(ev, session, laps=True, telemetry=False, messages=True, weather=False)

        data = await stats.format_results(s, session, year)

        table, ax = stats.results_table(data, session)
        ax.set_title(
            f"{ev['EventDate'].year} {ev['EventName']} - {session}"
        ).set_fontsize(13)

        f = utils.plot_to_file(table, "plot")
        embed = discord.Embed(
            title=f"{ev['EventDate'].year} {ev['EventName']} - {session}", color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

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
                await ctx.respond("Quiz setup is not configured for this server. Use /quizsetup.", ephemeral=True)
                return

            # Check if the user has any of the specific roles
            has_specific_role = any(
                role.id in role_ids for role in ctx.author.roles)

            if not has_specific_role:
                await ctx.respond("You don't have permissions to create a quiz.", ephemeral=True)
                return

            try:
                channel = ctx.guild.get_channel(channel_id)

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
                                embed.set_image(
                                    url="attachment://quiz_image.png")
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
                await ctx.respond("Quiz Created!", ephemeral=True)
                future_time = int(time.time()) + 20
                embed.add_field(name="Time Remaining",
                                value=f"<t:{future_time}:R>", inline=False)
                await asyncio.sleep(10)
                if image:
                    img_data = await image.read()
                    img = discord.File(io.BytesIO(img_data),
                                       filename="quiz_image.png")
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
                d = {"1️⃣": option1, "2️⃣": option2,
                     "3️⃣": option3, "4️⃣": option4}
                if correct_answers == 0:
                    winner_embed.description = f"No one answered correctly. The correct answer was **||{d.get(answer)}. {fact}||**"
                else:
                    winner_embed.description = f"The correct answer was **||{d.get(answer)}. {fact}||**. {correct_answers} {'person' if correct_answers == 1 else 'people'} answered correctly:"
                    for i, winner in enumerate(winners):
                        if i < 5:  # Mention up to 5 winners
                            winner_embed.description += f"\n{i + 1}. {winner.mention}"
                await quiz_message.reply(embed=winner_embed)
            except discord.Forbidden:
                await ctx.respond(f"I don't have enough permissions to send quizzes to <#{channel_id}>. I require media permissions for that!", ephemeral=True)
        else:
            await ctx.respond("Install this bot in your server to access quiz features!", ephemeral=True)

    @commands.slash_command(name="reddit", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def reddit(self, ctx: ApplicationContext):
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
        await ctx.respond(embed=embed, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(description="Race pitstops ranked by duration or filtered to a driver.", name="pitstops", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def pitstops(self, ctx: ApplicationContext, year: options.SeasonOption4, round: options.RoundOption,
                       filter: options.RankedPitstopFilter, driver: options.DriverOption):
        """Display pitstops for the race ranked by `filter` or `driver`.

        All parameters are optional. Defaults to the best pitstop per driver for the most recent race.
        Pitstop data unavailable before 2012.

        Usage:
        ----------
            /pitstops-ranked [season] [round] [filter]
        """

        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        # Get event info to match race name idenfifiers from command
        event = await stats.to_event(year, round)
        yr, rd = event["EventDate"].year, event["RoundNumber"]

        try:
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
            await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))
        except:
            await ctx.respond("Pitstop data for this event isn't available at the moment. Please try again later.")

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
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(
        description="View fastest sectors and speed trap based on quick laps. Seasons >= 2018.", integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        })
    async def sectors(self, ctx: ApplicationContext, year: options.SeasonOption3,
                      round: options.RoundOption, tyre: options.TyreOption, session: options.SessionOption):
        """View min sector times and max speedtrap per driver. Based on recorded quicklaps only."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        ev = await stats.to_event(year, round)
        await utils.check_season(ctx, year)
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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(description="Career stats for a driver. Enter Full Name of the driver.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def career(self, ctx: ApplicationContext, driver: options.driveroption2):

        loop = asyncio.get_running_loop()
        result_embed = await loop.run_in_executor(None, get_driver, driver, ctx)
        # send final embed
        await ctx.respond(embed=result_embed, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(
        name="track-incidents",
        description="Summary of race events including Safety Cars and retirements.", integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        })
    async def track_incidents(self, ctx: ApplicationContext,
                              year: options.SeasonOption3, round: options.RoundOption):
        """Outputs a table showing the lap number and event, such as Safety Car or Red Flag."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))


def setup(bot: discord.Bot):
    bot.add_cog(Race(bot))
