from f1.api.stats import roundnumber
import asyncio
import logging
import discord
from discord import ApplicationContext, Embed, default_permissions
from discord.ext import commands
from fastf1 import Cache as ff1_cache
from f1.api import fetch
from f1.config import Config
from f1.api.stats import get_top_role_color
from f1 import options
from f1.api import stats
import pandas as pd
from f1.api.stats import get_ephemeral_setting
from f1.options import resolve_years_fastf1, resolve_rounds
import sqlite3
import time
START_TIME = time.time()
logger = logging.getLogger("f1-bot")


class Admin(commands.Cog, guild_ids=Config().guilds):
    """Commands to manage the bot and view info."""

    def __init__(self, bot: discord.Bot):
        self.bot = bot

    async def _enable_cache(self, minutes: int):
        await asyncio.sleep(float(minutes * 60))
        fetch.use_cache = True
        ff1_cache.set_enabled()
        logger.warning("Cache re-enabled after timeout")

    def get_uptime(self):
        """Get running time since bot started. Return tuple (days, hours, minutes)."""
        invoke_time = time.time()
        uptime = invoke_time - START_TIME
        days, rem = divmod(uptime, 86400)
        hours, rem = divmod(rem, 3600)
        mins, secs = divmod(rem, 60)
        return (int(days), int(hours), int(mins), int(secs))

    @commands.slash_command(description="List of commands.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def help(self, ctx: ApplicationContext):
        from discord.ext.pages import Page, Paginator, PaginatorButton

        emd1 = Embed(
            title="Type `/command` and choose parameters",
            description="List of Commands",
            colour=get_top_role_color(ctx.author)


        )
        emd = Embed(
            title="",

            colour=get_top_role_color(ctx.author))
        emd1.add_field(
            name="/calendar",
            value="Provides season calendar."
        )

        emd1.add_field(
            name="/career",
            value="Career stats for a driver. Enter Full Name of the driver."
        )

        emd1.add_field(
            name="/fiadoc",
            value="Get latest FIA Document."
        )

        emd1.add_field(
            name="/constructors",
            value="Find constructors info"
        )
        emd1.add_field(
            name="/grid",
            value="Find all the participating drivers and teams for the given season."
        )
        emd1.add_field(
            name="/info",
            value="Find Bot info."
        )
        emd1.add_field(
            name="/laptimes",
            value="Best ranked lap times per driver."
        )

        emd1.add_field(
            name="/pitstops",
            value="Race pitstops ranked by duration or filtered to a driver."
        )

        emd1.add_field(
            name="/avg-lap-delta",
            value="Bar chart comparing average time per driver with overall race average as a delta."
        )

        emd1.add_field(
            name="/avgpos",
            value="Average position of a driver or a team in a span of season."
        )

        emd1.add_field(
            name="/cornering",
            value="Cornering Comparison of any two drivers."
        )

        emd1.add_field(
            name="/driver-lap-time-distribution",
            value="View driver(s) lap time distribution on track."
        )

        emd1.add_field(
            name="/fastestlaps",
            value="Show a bar chart comparing fastest laps in the session."
        )

        emd1.add_field(
            name="/gains",
            value="Show the position gains/losses per driver in the race."
        )

        emd1.add_field(
            name="/gear",
            value="Plot which gear is being used at which point of the track"
        )

        emd1.add_field(
            name="/h2h",
            value="Head to Head stats."
        )

        emd1.add_field(
            name="/lap-compare",
            value="Compare lap time difference between two drivers."
        )

        emd1.add_field(
            name="/lap-distribution",
            value="Violin plot comparing distribution of lap times on different types."
        )
        emd1.add_field(
            name="/track-evolution",
            value="Trackside weather data."
        )
        emd1.add_field(
            name="/quiz",
            value="Make Quizzes."
        )
        emd1.add_field(
            name="/leaderboard",
            value="Leaderboard for quizzes.")
        emd1.add_field(
            name="/quizsetup",
            value="Setup quiz for your server. (admin only)"
        )
        emd1.add_field(
            name='/technical-glossary',
            value="Search for a technical term.")
        emd.add_field(
            name="/positionchanges",
            value="Plot driver position changes in the race."
        )
        emd1.add_field(
            name="/circuits",
            value="View circuit info for the given season.")

        emd.add_field(
            name="/race-trace",
            value="Lap Comparison of participating drivers."
        )

        emd.add_field(
            name="/standing-history",
            value="Standing History of either WDC or WCC."
        )

        emd.add_field(
            name="/standings-heatmap",
            value="Plot WDC standings on a heatmap."
        )

        emd.add_field(
            name="/team-pace-delta",
            value="Rank team's race pace from the fastest to the slowest."
        )

        emd.add_field(
            name="/telemetry",
            value="Compare fastest lap telemetry between two drivers."
        )

        emd.add_field(
            name="/track-sectors",
            value="Compare fastest driver sectors on track map."
        )

        emd.add_field(
            name="/track-speed",
            value="View driver speed on track."
        )

        emd.add_field(
            name="/tyre-choice",
            value="Percentage distribution of tyre compounds."
        )

        emd.add_field(
            name="/tyre-degradation",
            value="Plot the performance of each tyre compound based on the age of the tyre."
        )

        emd.add_field(
            name="/tyre-strats",
            value="Tyre Strategies of the drivers' in a race."
        )
        emd1.add_field(
            name="/track-elevation",
            value="View circuit's 3D visualisation.")

        emd.add_field(
            name="/results",
            value="Result data for the session. Default last race."
        )

        emd.add_field(
            name="/schedule",
            value="Get race schedule"
        )

        emd.add_field(
            name="/sectors",
            value="View fastest sectors and speed trap based on quick laps."
        )

        emd.add_field(
            name="/wcc",
            value="Constructors Championship standings."
        )

        emd.add_field(
            name="/wdc",
            value="Driver championship standings."
        )

        emd.add_field(
            name="/wdc-contenders",
            value="Shows a list of drivers who can still mathematically win the wdc."
        )
        emd.add_field(
            name="/track-incidents",
            value="Shows a table showing the lap number and event, such as Safety Car or Red Flag."
        )

        emd.add_field(
            name="/radio",
            value="Shows Team Radio"
        )

        emd.add_field(
            name="/racecontrol",
            value="Shows Race Control Data."
        )
        emd.add_field(
            name="/meme",
            value="Shows a random f1 meme. (from r/formuladank)"
        )
        emd.add_field(
            name="/regulations",
            value="View formula 1 regulations.")
        emd.add_field(
            name="/records",
            value="View formula 1 records.")

        emd.add_field(
            name="/generate-cache",
            value="Generates cache for a given f1 race weekend and thus, speeds up some of the plotting functions.")
        emd.add_field(
            name="/silent-mode",
            value="Makes the messages visible only to the user who issued the command. (admin only")

        embeds = [Page(embeds=[emd1]), Page(embeds=[emd])]
        buttons = [
            PaginatorButton("first", label="First",
                            style=discord.ButtonStyle.blurple),
            PaginatorButton("prev", label="Previous",
                            style=discord.ButtonStyle.red),
            PaginatorButton("page_indicator",
                            style=discord.ButtonStyle.gray, disabled=True),
            PaginatorButton("next", label="Next",
                            style=discord.ButtonStyle.green),
            PaginatorButton("last", label="Last",
                            style=discord.ButtonStyle.blurple),
        ]

        paginator = Paginator(pages=embeds, timeout=898, author_check=False, show_indicator=True, use_default_buttons=False,
                              custom_buttons=buttons)
        try:
            await paginator.respond(ctx.interaction, ephemeral=get_ephemeral_setting(ctx))
        except discord.Forbidden:
            return
        except discord.HTTPException:
            return

    @commands.slash_command(name="generate-cache", description="Generate cache for a given f1 race weekend to speed up plotting.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def cache(self,
                    ctx: ApplicationContext, year: discord.Option(int, "Select the season", default=None, autocomplete=resolve_years_fastf1),
                    round: discord.Option(str, "Select the round (event)", default=None, autocomplete=resolve_rounds)):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await ctx.respond("Generating cache for the given event", ephemeral=True)
        event = await stats.to_event(year, round)

        # Run session loading concurrently
        tasks = [
            stats.load_session(event, i, laps=True, telemetry=True, weather=True, messages=True)
            for i in range(5, 0, -1)
        ]

        await asyncio.gather(*tasks)


    @commands.slash_command(name="silent-mode", description="Makes the messages visible only to the user who issued the command.")
    @default_permissions(administrator=True)
    async def ephemeral(self,
                        ctx: ApplicationContext,
                        silent_mode: options.EphemeralOption):
        if ctx.guild.name is not None:
            conn = sqlite3.connect("bot_settings.db")
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                guild_id INTEGER PRIMARY KEY,
                ephemeral_setting BOOLEAN NOT NULL
            )
            """)
            conn.commit()
            conn.close()
            guild_id = ctx.guild.id

    # Update the database with the setting
            conn = sqlite3.connect("bot_settings.db")
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO settings (guild_id, ephemeral_setting)
            VALUES (?, ?)
            ON CONFLICT(guild_id) DO UPDATE SET ephemeral_setting = excluded.ephemeral_setting
            """, (guild_id, silent_mode))

            conn.commit()
            conn.close()

            await ctx.respond("Settings updated for this server!", ephemeral=True)
        else:
            await ctx.respond("You can only use this command after installing it on your server!", ephemeral=True)

    @commands.slash_command(description="Bot information and status.")
    async def info(self, ctx: ApplicationContext):
        uptime = self.get_uptime()
        app_info = await self.bot.application_info()
        latency = int(self.bot.latency * 1000)

        if self.bot.is_closed():
            ws = "```diff\n- Closed\n```"
        else:
            ws = "```diff\n+ Open\n```"

        emd = Embed(
            title=f"{app_info.name}",
            description="formulaStats is a discord bot implementation to view Formula 1 statistics and other visuals via slash commands. formulaStats sources its data from [FastF1](https://github.com/theOehrly/Fast-F1) and [Jolpica (Now deprecated Ergast's successor)](https://github.com/jolpica/jolpica-f1), and uses [Pycord](https://github.com/Pycord-Development/pycord) to interact with the discord API.", color=get_top_role_color(ctx.author))
        emd.add_field(
            name="Github repository", value="https://github.com/Carol52x/formulaStats", inline=False)
        emd.add_field(
            name="Uptime", value=f"{uptime[0]}d, {uptime[1]}h, {uptime[2]}m", inline=True)
        emd.add_field(name="Ping", value=f"{latency} ms", inline=True)
        emd.add_field(name="Connection", value=ws, inline=True)
        emd.set_footer(
            text='Bug reports can be reported by contacting carol520 or opening a github issue.')
        await ctx.respond(embed=emd, ephemeral=get_ephemeral_setting(ctx))


def setup(bot: discord.Bot):
    bot.add_cog(Admin(bot))
