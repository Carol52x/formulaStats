from datetime import datetime, date
from f1.api.stats import get_top_role_color
from f1.utils import check_season
from f1.config import Config
from f1.api import ergast, stats
from f1 import utils
from f1 import options
import logging
import discord
import fastf1
import pandas as pd
from datetime import date
from f1.api.stats import get_ephemeral_setting
from discord.ext import commands
import asyncio
from f1.api.stats import schedule, get_fia_doc
import discord
import pandas as pd
import typing
from discord.ext import commands
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"
fastf1.Cache.enable_cache('cache/')
logger = logging.getLogger("f1-bot")


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
        await ctx.respond(file=f, embed=embed, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(description="Provides season calender", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def calender(self, ctx, year: options.SeasonOption2):
        if year == None:
            year = int(datetime.now().year)
        await check_season(ctx, year)

        calender = await ergast.get_race_schedule2(year)
        table, ax = stats.plot_race_schedule(calender['data'], year)
        ax.set_title(f"{year} Calender").set_fontsize(12)
        f = utils.plot_to_file(table, "plot")

        embed = discord.Embed(
            title=f'Calender {year}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        if year > 2020:
            embed.set_footer(text="The highlighted events are Sprint weekends.")

        await ctx.respond(file=f, embed=embed, ephemeral=get_ephemeral_setting(ctx))

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
        try:
            result = await ergast.get_team_standings(year, last_round)
        except IndexError:
            result = await ergast.get_team_standings(year, last_round-1)

        table, ax = stats.championship_table(result['data'], type="wcc")
        yr, rd = result['season'], result['round']
        ax.set_title(
            f"{yr} Constructor Championship - Round {rd}").set_fontsize(12)

        f = utils.plot_to_file(table, "plot")
        embed = discord.Embed(
            title=f'WCC standings for the {yr} season', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(file=f, embed=embed, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(description='get race schedule', integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def schedule(self, ctx):
        loop = asyncio.get_running_loop()
        message_embed = await loop.run_in_executor(None, schedule, ctx)
        if type(message_embed) == str:
            await ctx.respond(message_embed, ephemeral=get_ephemeral_setting(ctx))
        else:
            await ctx.respond(embed=message_embed, ephemeral=get_ephemeral_setting(ctx))



def setup(bot: discord.Bot):
    bot.add_cog(Season(bot))
