
import asyncio
import logging
import sys
import time
import requests
import discord
from discord import ApplicationContext, Embed, default_permissions
from discord.ext import commands
from fastf1 import Cache as ff1_cache
from io import BytesIO

from f1 import utils
from f1.api import fetch
from f1.api.ergast import check_status
from f1.config import Config
from f1.target import MessageTarget
from f1.api.stats import get_top_role_color
logger = logging.getLogger("f1-bot")

# set global time bot started
START_TIME = time.time()


class Admin(commands.Cog, guild_ids=Config().guilds):
    """Commands to manage the bot and view info."""

    def __init__(self, bot: discord.Bot):
        self.bot = bot

    

    def get_uptime(self):
        """Get running time since bot started. Return tuple (days, hours, minutes)."""
        invoke_time = time.time()
        uptime = invoke_time - START_TIME
        days, rem = divmod(uptime, 86400)
        hours, rem = divmod(rem, 3600)
        mins, secs = divmod(rem, 60)
        return (int(days), int(hours), int(mins), int(secs))

    async def _enable_cache(self, minutes: int):
        await asyncio.sleep(float(minutes * 60))
        fetch.use_cache = True
        ff1_cache.set_enabled()
        logger.warning("Cache re-enabled after timeout")

    @commands.slash_command(integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def help(self, ctx: ApplicationContext):

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
            value="Provides season calendar"
        )

        emd1.add_field(
            name="/career",
            value="Career stats for a driver. Enter Full Name of the driver."
        )

        emd1.add_field(
            name="/fiadoc",
            value="Get latest FIA Document. Use 2, 3… for older documents."
        )

        emd1.add_field(
            name="/help",
            value="No description provided"
        )

        emd1.add_field(
            name="/info",
            value="Bot information and status."
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
            name="/avg pos",
            value="Average position of a driver or a team in a span of season. (This may take about a minute or two.)"
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
            name="/gap",
            value="Plots the delta in seconds between two drivers over a lap."
        )

        emd1.add_field(
            name="/gear",
            value="Plot which gear is being used at which point of the track"
        )

        emd1.add_field(
            name="/h2h",
            value="Head to Head stats. (This may take about a minute or two.)"
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
            value="Make Quizzes"
        )
        emd1.add_field(
            name="/quizsetup",
            value="Setup quiz for your server."
        )
        emd.add_field(
            name="/positionchanges",
            value="Plot driver position changes in the race."
        )

        emd.add_field(
            name="/race-trace",
            value="Lap Comparison of participating drivers"
        )

        emd.add_field(
            name="/speed-comparison",
            value="Speed Comparison (Time or Distance) of any two drivers."
        )

        emd.add_field(
            name="/standing-history",
            value="Standing History of either WDC or WCC"
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
            name="/tyre-performance",
            value="Plot the performance of each tyre compound based on the age of the tyre."
        )

        emd.add_field(
            name="/tyre-strats",
            value="Tyre Strategies of the drivers' in a race."
        )

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
            value="View fastest sectors and speed trap based on quick laps. Seasons >= 2018."
        )

        emd.add_field(
            name="/track-incidents",
            value="Summary of race events including Safety Cars and retirements."
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
            name="/radio",
            value="Shows Team Radio"
        )

        emd.add_field(
            name="/racecontrol",
            value="Shows Race Control Data"
        )
        emd.add_field(
            name="/reddit",
            value="Shows a random post from r/formuladank"
        )

        embeds = [emd1, emd]

# Send both embeds together in a single message
        await ctx.respond(embeds=embeds, ephemeral=True)


def setup(bot: discord.Bot):
    bot.add_cog(Admin(bot))
