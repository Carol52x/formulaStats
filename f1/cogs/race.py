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
import json
from datetime import datetime
import subprocess
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
    async def radio(self, ctx: ApplicationContext, driver: options.DriverOptionRequired(), round: options.RoundOption, session: options.SessionOption, year: options.SeasonOption3):
        from discord.ext.pages import Page, PaginatorButton, Paginator
        import requests
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        try:

            if str(round).isdigit():
                round = int(round)
            session = stats.convert_shootout_to_qualifying(year, session)

            ev = await stats.to_event(year, round)
            session1 = await stats.load_session(ev, session, telemetry=False, laps=False, weather=False, messages=False)

            if not driver.isdigit():
                driver_number = session1.get_driver(
                    driver[0:3].upper())['DriverNumber']
            else:
                driver_number = driver
            # fallback for driver headshot url as data seems to be unavailable for the 2018 season.
            if year == 2018:
                ev2 = await stats.to_event(year+1, round)
                session2 = await stats.load_session(ev2, session, telemetry=False, laps=False, weather=False, messages=False)
                driver_url = session2.get_driver(
                    driver[0:3].upper())['HeadshotUrl']
            elif year == 2019 and round < 7:  # fallback for handling missing data from the 2019 season as well
                ev2 = await stats.to_event(year+1, round+6)
                session2 = await stats.load_session(ev2, session, telemetry=False, laps=False, weather=False, messages=False)
                driver_url = session2.get_driver(
                    driver[0:3].upper())['HeadshotUrl']
            else:
                driver_url = session1.get_driver(
                    driver[0:3].upper())['HeadshotUrl']

            path = "https://livetiming.formula1.com" + session1.api_path + "TeamRadio.json"
            response = requests.get(path)
            radio_data = json.loads(response.content.decode("utf-8-sig"))
            data = [entry for entry in radio_data["Captures"]
                    if entry["RacingNumber"] == str(driver_number)]
            urls = []
            for i in data:
                urls.append("https://livetiming.formula1.com" +
                            session1.api_path + i['Path'])
            length = len(urls)

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
                    return discord.File(video_file, filename="output.mp4")

            class RadioPaginator(PaginatorButton):
                def __init__(self, button_type, label=None, style=None, **kwargs):
                    super().__init__(button_type, label=label, style=style, **kwargs)

                async def callback(self, interaction):
                    new_page = self.paginator.current_page
                    if self.button_type == "first":
                        new_page = 0
                    elif self.button_type == "prev":
                        if self.paginator.loop_pages and self.paginator.current_page == 0:
                            new_page = self.paginator.page_count
                        else:
                            new_page -= 1
                    elif self.button_type == "next":
                        if (
                            self.paginator.loop_pages
                            and self.paginator.current_page == self.paginator.page_count
                        ):
                            new_page = 0
                        else:
                            new_page += 1
                    elif self.button_type == "last":
                        new_page = self.paginator.page_count
                    await self.paginator.goto_page(page_number=new_page, interaction=interaction)
                    page_number = self.paginator.current_page

                    value = page_number

                    if value is not None:

                        mp3_url = urls[value]
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

                        # Create the video from the MP3 and image
                        create_video(
                            mp3_bytes, image_bytes, output_file)

                        with open(output_file, "rb") as video_file:
                            await interaction.edit(file=discord.File(video_file, filename="output.mp4"))
                    await send_video(mp3_url, driver_url)
            mypage = []
            mp3_url_1 = urls[0]
            file_1 = await send_video(ctx, mp3_url_1, driver_url)

            for i in range(0, int(length)):

                embed = discord.Embed(
                    title=f"{ev['EventDate'].year} {ev['EventName']}- Radios for {session1.get_driver(driver[0:3].upper())['LastName']}",
                    color=get_top_role_color(ctx.author)
                )
                embed.set_image(url=f"attachment://plot.png")
                mypage.append(Page(embeds=[embed], files=[file_1]))
            buttons = [
                RadioPaginator("first", label="<<",
                               style=discord.ButtonStyle.blurple),
                RadioPaginator("prev", label="<",
                               style=discord.ButtonStyle.red),
                RadioPaginator("page_indicator",
                               style=discord.ButtonStyle.gray, disabled=True),
                RadioPaginator("next", label=">",
                               style=discord.ButtonStyle.green),
                RadioPaginator("last", label=">>",
                               style=discord.ButtonStyle.blurple),
            ]

            paginator = Paginator(pages=mypage, timeout=None, author_check=False, show_indicator=True, use_default_buttons=False,
                                  custom_buttons=buttons)
            try:
                await paginator.respond(ctx.interaction)
            except discord.Forbidden:
                return
            except discord.HTTPException:
                return
        except IndexError:
            await ctx.respond("No data for the driver found.")
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
        session = stats.convert_shootout_to_qualifying(year, session)

        ev = await stats.to_event(year, round)
        Session = await stats.load_session(ev, session, messages=True, laps=True, telemetry=True, weather=True)

        messages = await asyncio.to_thread(lambda: Session.race_control_messages)

        files = stats.racecontrol(messages, session)

        mypage = []
        for i in files:

            file = utils.plot_to_file(i, f"plot")
            embed = discord.Embed(
                title=f"{ev['EventDate'].year} {ev['EventName']} {session} - Race Control",
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
            try:
                table = stats.stints_driver(stints)
            except KeyError:
                await ctx.respond("No data for that driver found.")
                return
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
    async def whocanwinwdc(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        try:
            loop = asyncio.get_running_loop()
            driver_standings = await loop.run_in_executor(None, get_drivers_standings, round, year)

    # Get the maximum amount of points
            points = await loop.run_in_executor(None, calculate_max_points_for_remaining_season, round, year)

            contenders = []

        # Iterate over the generator object and extract the relevant information
            for contender in await loop.run_in_executor(None, calculate_who_can_win, driver_standings, points):
                contenders.append(contender)
            a = stats.plot_chances(contenders)
            f = utils.plot_to_file(a, "plot")
            embed = discord.Embed(title=f'Theoretical WDC Contenders: {year} Round: {round}',
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
        session = stats.convert_shootout_to_qualifying(year, session)
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
                await ctx.respond("Quiz Created!", ephemeral=True)
                await channel.send(embed=embed2, delete_after=10)
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
                conn = sqlite3.connect("leaderboard.db")
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS winners (
                    guild_id INTEGER,
                    user_id INTEGER,
                    wins INTEGER,
                    PRIMARY KEY (guild_id, user_id)
                )
                """)

                conn.commit()
                if correct_answers == 0:
                    winner_embed.description = f"No one answered correctly. The correct answer was **||{d.get(answer)}. {fact}||**"
                else:
                    winner_embed.description = f"The correct answer was **||{d.get(answer)}. {fact}||**. {correct_answers} {'person' if correct_answers == 1 else 'people'} answered correctly:"
                    for i, winner in enumerate(winners):
                        cursor.execute("""
                        INSERT INTO winners (guild_id, user_id, wins)
                        VALUES (?, ?, 1)
                        ON CONFLICT(guild_id, user_id)
                        DO UPDATE SET wins = wins + 1
                        """, (guild_id, winner.id))
                        conn.commit()
                        if i < 5:  # Mention up to 5 winners
                            winner_embed.description += f"\n{i + 1}. {winner.mention}"

                await quiz_message.reply(embed=winner_embed)
            except discord.Forbidden:
                await ctx.respond(f"I don't have enough permissions to send quizzes to <#{channel_id}>. I require media permissions for that!", ephemeral=True)
        else:
            await ctx.respond("Install this bot in your server to access quiz features!", ephemeral=True)

    @commands.slash_command(name="leaderboard", description="View quiz leaderboard.")
    async def leaderboard(self, ctx: ApplicationContext):
        conn = sqlite3.connect("leaderboard.db")
        cursor = conn.cursor()
        guild_id = ctx.guild.id
        user_id = ctx.author.id
        cursor.execute("""
        SELECT user_id, wins
        FROM winners
        WHERE guild_id = ?
        ORDER BY wins DESC
        """, (guild_id,))
        all_users = cursor.fetchall()

        if not all_users:
            await ctx.respond("No leaderboard data available for this server.", ephemeral=True)
            return

        # Determine the rank of the user invoking the command
        cursor.execute("""
        SELECT RANK
        FROM (
            SELECT user_id, RANK() OVER (ORDER BY wins DESC) AS RANK
            FROM winners
            WHERE guild_id = ?
        ) WHERE user_id = ?
        """, (guild_id, user_id))
        user_rank = cursor.fetchone()
        user_rank_text = f"Your rank: #{user_rank[0]}" if user_rank else "Your rank: No data (you haven't participated yet!)"
        pages = []
        from discord.ext.pages import Page, Paginator
        for i in range(0, len(all_users), 10):  # Paginate 10 users per page
            embed = discord.Embed(
                title=f"Quiz Leaderboard for {ctx.guild.name}",
                color=get_top_role_color(ctx.author)
            )

            for rank, (user_id, wins) in enumerate(all_users[i:i + 10], start=i + 1):
                user = ctx.guild.get_member(user_id)
                username = user.name if user else f"User ID: {user_id}"
                embed.add_field(name=f"#{rank} {username}",
                                value=f"Wins: {wins}", inline=False)

            embed.set_footer(text=user_rank_text)
            pages.append(Page(embeds=[embed]))

        paginator = Paginator(pages=pages, timeout=899, author_check=False)
        try:
            await paginator.respond(ctx.interaction, ephemeral=get_ephemeral_setting(ctx))
        except discord.Forbidden:
            return
        except discord.HTTPException:
            return

    @commands.slash_command(name="meme", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def meme(self, ctx: ApplicationContext):
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
            if year > 2017:
                s = await stats.load_session(event, "R", laps=True, telemetry=True)
                data = await stats.filter_pitstops(yr, rd, s, filter, driver)
                text = "The stationary times are an appromixation and thus, may be anomalous."
            else:
                data = await stats.filter_pitstops(yr, rd, filter=filter, driver=driver)
                text = ""

            table, ax = stats.pitstops_table(data, year)
            ax.set_title(
                f"{yr} {event['EventName']} | Pitstops ({filter})"
            ).set_fontsize(13)

            f = utils.plot_to_file(table, "plot")
            embed = discord.Embed(
                title=f"{yr} {event['EventName']} | Pitstops", color=get_top_role_color(ctx.author))
            embed.set_image(url="attachment://plot.png")

            embed.set_footer(text=text)
            await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))
        except IndexError:
            if driver:
                await ctx.respond("No data for the driver found.")
                return
            else:
                pass
        except:
            await ctx.respond("Pitstop data for this session is not available yet.", ephemeral=get_ephemeral_setting(ctx))

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
        session = stats.convert_shootout_to_qualifying(year, session)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True)
        data = await stats.fastest_laps(s, tyre)
        if not year == 2018:
            absolute_compounds = await stats.get_compound_async(year, event.EventName)
            compound_numbers = [int(s[1:]) for s in absolute_compounds]
            absolute_number_mapping = {i: j for i, j in zip(
                compound_numbers, absolute_compounds)}
            soft_compound = absolute_number_mapping.get(max(compound_numbers))
            hard_compound = absolute_number_mapping.get(min(compound_numbers))
            remaining_compound = next(compound for compound, number in absolute_number_mapping.items()
                                      if number != soft_compound and number != hard_compound)
            compound_label_mapping = {
                "SOFT": soft_compound,
                "MEDIUM": absolute_number_mapping.get(remaining_compound),
                "HARD": hard_compound
            }
            data["Tyre"] = data["Tyre"].apply(
                lambda x: x + " " + compound_label_mapping.get(x, ""))
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
        session = stats.convert_shootout_to_qualifying(year, session)
        yr, rd = ev["EventDate"].year, ev["RoundNumber"]
        s = await stats.load_session(ev, session, laps=True)
        data = await stats.sectors(s, tyre)

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

    @commands.slash_command(description="Constructor information for the given season", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def constructors(self, ctx: ApplicationContext, year: options.SeasonOption):
        year = roundnumber(round=None, year=year)[1]
        await utils.check_season(ctx, year)

        url = f'https://api.jolpi.ca/ergast/f1/{year}/constructors/'
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                standings = await response.json()
        if standings:
            results = {
                'season': standings['MRData']['ConstructorTable']['season'],
                'data': [],
            }
            for standing in standings['MRData']['ConstructorTable']['Constructors']:
                results['data'].append(
                    {
                        'Constructor': standing['name'],
                        'Nationality': standing['nationality'],
                    }
                )
        table, ax = stats.constructor_table(results['data'])
        ax.set_title(f"{year} Formula 1 Teams").set_fontsize(12)
        embed = discord.Embed(
            title=f'Teams participating in the {year} season', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        f = utils.plot_to_file(table, f"plot")
        await ctx.respond(file=f, embed=embed, ephemeral=get_ephemeral_setting(ctx))

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
        s = await stats.load_session(ev, 'Race', laps=True, messages=True)
        dnfs = stats.get_dnf_results(s)
        # Combine driver code and dnf reason into single column for merging
        dnfs["Event"] = dnfs.apply(
            lambda row: f"Retired: {row['Abbreviation']} ({row['Status']})", axis=1)
        dnfs = dnfs.loc[:, ["LapNumber", "Event"]]
        # Get track status events grouped by lap number
        track_events = stats.get_track_events(s)
        if dnfs["Event"].size == 0 and track_events["Event"].size == 0:
            raise MissingDataError("No track events in this race.")
        incidents = pd.concat([
            dnfs.loc[:, ["LapNumber", "Event"]],
            track_events.loc[:, ["LapNumber", "Event"]]
        ], ignore_index=True).sort_values(by="LapNumber").reset_index(drop=True)
        incidents.dropna(inplace=True)
        incidents["LapNumber"] = incidents["LapNumber"]
        table, ax = stats.incidents_table(incidents)
        ax.set_title(
            f"{ev['EventDate'].year} {ev['EventName']}\nTrack Incidents"
        ).set_fontsize(12)
        f = utils.plot_to_file(
            table, f"plot")
        embed = discord.Embed(title=f'Track incidents: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(
        name="technical-glossary",
        description="Short description of a technical term.", integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        })
    async def technical(self, ctx: ApplicationContext,
                        term: str):
        from bs4 import BeautifulSoup
        import requests
        from rapidfuzz import fuzz

        url = f"https://www.f1technical.net/glossary/{term[0].lower()}"
        html = await asyncio.to_thread(lambda: requests.get(url=url))
        soup = BeautifulSoup(html.content, 'html.parser')
        glossary = {}

        for dt in soup.find_all('dt'):
            dd = dt.find_next_sibling('dd')
            if dd:
                glossary[dt.get_text(strip=True)] = dd.get_text(strip=True)

        # Function to find the best match manually
        def find_best_match(term, keys):
            best_match = None
            highest_score = 0
            for key in keys:
                score = fuzz.ratio(term, key)
                if score > highest_score:
                    highest_score = score
                    best_match = key
            return best_match, highest_score

        # Attempt to match the original term
        if glossary:
            matched_title, score = find_best_match(term, glossary.keys())
            if matched_title and score >= 70:
                embed = discord.Embed(title=matched_title,
                                      description=glossary[matched_title],
                                      color=get_top_role_color(ctx.author))
                embed.set_footer(text="Source: f1technical.net")
                await ctx.respond(embed=embed, ephemeral=get_ephemeral_setting(ctx))
                return
            else:
                for key in glossary.keys():
                    if '(' in key and ')' in key:
                        parenthetical_text = key[key.find(
                            '(') + 1:key.find(')')].strip()
                        score = fuzz.ratio(
                            term.lower(), parenthetical_text.lower())
                        if score >= 100:
                            embed = discord.Embed(title=key,
                                                  description=glossary[key],
                                                  color=get_top_role_color(ctx.author))
                            embed.set_footer(text="Source: f1technical.net")
                            await ctx.respond(embed=embed, ephemeral=get_ephemeral_setting(ctx))
                            return
        await ctx.respond("Given term not found.", ephemeral=get_ephemeral_setting(ctx))


def setup(bot: discord.Bot):
    bot.add_cog(Race(bot))
