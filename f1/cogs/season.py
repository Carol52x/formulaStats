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
import wikipedia
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

        calender = await ergast.get_race_schedule(year)
        table, ax = stats.plot_race_schedule(calender['data'], year)
        ax.set_title(f"{year} Calender").set_fontsize(12)
        f = utils.plot_to_file(table, "plot")

        embed = discord.Embed(
            title=f'Calender {year}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        if year > 2020:
            embed.set_footer(
                text="The highlighted events are Sprint weekends.")

        await ctx.respond(file=f, embed=embed, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name='fiadoc', description='Get FIA documents.', integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def fiadoc(self, ctx, doc: options.DocumentOption, year: options.SeasonOption5, round: options.RoundOption,
                     get_all_docs: options.DocumentOption2):

        from discord.ext.pages import Paginator, Page
        year = stats.roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        if year == int(datetime.now().year) and round is None:
            eventname = None
            round = stats.roundnumber(round, year)[0]
        else:
            round = stats.roundnumber(round, year)[0]
            event = await stats.to_event(year, round)
            eventname = event.EventName
        mypage = []
        loop = asyncio.get_running_loop()

        if eventname is None:
            event = await stats.to_event(year, round)
            eventname = event.EventName
        if get_all_docs is True:
            options_list = await loop.run_in_executor(None, get_fia_doc, year, eventname, doc, None, True)

            def truncate_before_gp_name(path):
                import re
                result = re.split(r'Grand Prix - ', path, maxsplit=1)
                if len(result) == 2:
                    return result[1]
                return path
            options_mapping = {truncate_before_gp_name(
                path): path for path in options_list}
            embed = discord.Embed(
                title="Choose your document:", color=get_top_role_color(ctx.author))

            class SelectMenu(discord.ui.View):

                def __init__(self, options, timeout=None):
                    super().__init__(timeout=timeout)
                    self.menus = self.create_select_menus(options)

                    for menu in self.menus:
                        self.add_item(menu)

                def create_select_menus(self, options):
                    # Split options into chunks of 25 or less
                    MAX_OPTIONS = 25
                    option_chunks = [options[i:i + MAX_OPTIONS]
                                     for i in range(0, len(options), MAX_OPTIONS)]
                    select_menus = []

                    for idx, chunk in enumerate(option_chunks):
                        select_menus.append(discord.ui.Select(
                            placeholder=f"Choose a document (Menu {idx + 1})",
                            min_values=1,
                            max_values=1,
                            options=[discord.SelectOption(
                                label=option, value=option) for option in chunk]
                        ))

                    return select_menus

                # Define a callback for each select menu dynamically
                async def interaction_check(self, interaction):
                    selected_value = interaction.data['values'][0]
                    await self.handle_selection(interaction, selected_value)

                async def handle_selection(self, interaction, doc_name):
                    await interaction.response.send_message(f"Preparing the document...", ephemeral=True)
                    loop = asyncio.get_running_loop()
                    images = await loop.run_in_executor(None, get_fia_doc, year, eventname, doc, options_mapping.get(doc_name))
                    mypage = []

                    for idx, image in enumerate(images):
                        embed = discord.Embed(
                            title=f"FIA Document(s) for {year} {eventname}",
                            color=get_top_role_color(ctx.author)
                        ).set_thumbnail(
                            url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg/1200px-F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg.png'
                        ).set_image(url=f"attachment://{idx}.png")

                        mypage.append(Page(embeds=[embed], files=[image]))

                    paginator = Paginator(
                        pages=mypage, timeout=898, author_check=False)
                    try:
                        await paginator.respond(ctx.interaction, ephemeral=get_ephemeral_setting(ctx))
                    except discord.Forbidden:
                        return
                    except discord.HTTPException:
                        return
            options = list(options_mapping.keys())
            await ctx.respond(embed=embed, view=SelectMenu(options))
        else:
            try:
                images = await loop.run_in_executor(None, get_fia_doc, year, eventname, doc)
            except:
                await ctx.respond("No documents found for the given session.")
                return
            a = 0

            for i in images:

                embed = discord.Embed(title=f"FIA Document(s) for {year} {eventname}", description="", color=get_top_role_color(ctx.author)).set_thumbnail(
                    url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg/1200px-F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg.png')
                embed.set_image(url=f"attachment://{a}.png")

                mypage.append(Page(embeds=[embed], files=[i]))
                a += 1

            paginator = Paginator(
                pages=mypage, timeout=None, author_check=False)
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

    @commands.slash_command(description='Get F1 records', integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def records(self, ctx, type: options.RecordOption):
        from io import StringIO
        import requests
        from bs4 import BeautifulSoup
        from plottable import ColDef, Table
        if type == "Drivers" or type == "Other Driver records":
            url = "https://en.wikipedia.org/wiki/List_of_Formula_One_driver_records"
        elif type == "Constructors":
            url = "https://en.wikipedia.org/wiki/List_of_Formula_One_constructor_records"
        elif type == "Tyres":
            url = "https://en.wikipedia.org/wiki/Formula_One_tyres#Records"
        elif type == "Engines":
            url = "https://en.wikipedia.org/wiki/Formula_One_engines#Records"
        elif type == "Races":
            url = "https://en.wikipedia.org/wiki/List_of_Formula_One_race_records"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headings_list = await asyncio.to_thread(lambda: soup.find_all('h3'))
        headings = [heading.get_text(strip=True).replace(
            "[edit]", "") for heading in headings_list]

        class Menu(discord.ui.View):

            def __init__(self, options, timeout=None):
                super().__init__(timeout=timeout)
                self.menus = self.create_select_menus(options)

                for menu in self.menus:
                    self.add_item(menu)

            def create_select_menus(self, options):
                MAX_OPTIONS = 25
                option_chunks = [options[i:i + MAX_OPTIONS]
                                 for i in range(0, len(options), MAX_OPTIONS)]
                select_menus = []
                for idx, chunk in enumerate(option_chunks):
                    select_menus.append(discord.ui.Select(
                        placeholder=f"Menu {idx + 1}",
                        min_values=1,
                        max_values=1,
                        options=[discord.SelectOption(
                            label=option, value=option) for option in chunk]
                    ))

                return select_menus

            # Define a callback for each select menu dynamically
            async def interaction_check(self, interaction):
                selected_value = interaction.data['values'][0]
                await self.handle_selection(interaction, selected_value)

            async def handle_selection(self, interaction, value):
                for heading in headings_list:
                    text = heading.get_text(strip=True).replace("[edit]", "")

                    if text.lower() == value.lower():
                        name = text
                        sibling = heading
                        table = sibling.find_next('table', class_='wikitable')
                        for span in table.find_all('span', style=lambda value: value and 'display:none' in value):
                            span.decompose()
                        df = pd.read_html(str(table))[0]

                # Clean up column names
                df.columns = [
                    cl.replace(
                        " details", "") if "compound details" in cl.lower() else cl
                    for cl in df.columns
                ]
                df.columns = df.columns.str.replace(
                    r'\[failed verification\]', '', regex=True)

                import re
                df.fillna(" ", inplace=True)
                df.set_index(df.columns[0], inplace=True, drop=True)
                df.drop(df.index[-1], inplace=True)
                df = df.applymap(lambda x: re.sub(
                    r'\[.*?\]', '', str(x)).replace("/", "") if isinstance(x, str) else x)

                col_defs = []
                default_textprops = {"ha": "left"}

                df.drop(columns=[col for col in df.columns if col.lower(
                ).startswith("unnamed")], inplace=True)

                for idx, col in enumerate(df.columns):
                    base_width = 1.8  # Increased base width
                    max_length = max(df[col].apply(lambda x: len(str(x))))
                    col_name_length = len(col)  # Length of the column name
                    width = max(base_width, (col_name_length + max_length)
                                * 0.12)  # Adjust scaling factor
                    col_defs.append(
                        ColDef(col, width=width, textprops=default_textprops))

                num_rows, num_cols = df.shape

                # Dynamically adjust figsize with some padding for better scaling
                # Increased scaling factor for width and height
                figsize = (num_cols * 2, num_rows * 0.6)
                figsize = (figsize[0] + 2, figsize[1] + 2)  # Added padding

                index_name = df.columns[0]

                loop = asyncio.get_running_loop()
                fig = await loop.run_in_executor(None, stats.plot_table, df, col_defs, index_name, figsize)

                fig = fig.figure
                f = utils.plot_to_file(fig, f"plot")
                embed = discord.Embed(
                    url=url, title=name, color=get_top_role_color(ctx.author))
                embed.set_image(url="attachment://plot.png")
                await interaction.edit(embed=embed, file=f)
        embed = discord.Embed(url=url, title="Choose the record to view:")
        if headings != []:
            if type == "Other Driver records":
                headings = headings[119:125]
            else:
                headings = headings[:119]
            if type == "Engines":
                headings = headings[-3:]
            if type == "Tyres":
                table = await asyncio.to_thread(lambda: soup.find('table', class_='wikitable'))
                for span in table.find_all('span', style=lambda value: value and 'display:none' in value):
                    span.decompose()
                table = str(table)
                df = pd.read_html(StringIO(str(table)))[0]

                df.columns = [
                    cl.replace(
                        " details", "") if "compound details" in cl.lower() else cl
                    for cl in df.columns
                ]
                df.columns = df.columns.str.replace(
                    r'\[failed verification\]', '', regex=True)

                # Clean up NaN values and remove unwanted characters
                df.fillna(" ", inplace=True)
                df.set_index(df.columns[0], inplace=True, drop=False)
                df.drop(df.index[-1], inplace=True)
                df = df.applymap(lambda x: str(x).replace('[', '').replace(
                    ']', '').replace("/", "") if isinstance(x, str) else x)

                col_defs = []
                default_textprops = {"ha": "left"}

                df.drop(columns=[col for col in df.columns if col.lower(
                ).startswith("unnamed")], inplace=True)

                for idx, col in enumerate(df.columns):
                    base_width = 1.8  # Increased base width
                    max_length = max(df[col].apply(lambda x: len(str(x))))
                    col_name_length = len(col)  # Length of the column name
                    width = max(base_width, (col_name_length + max_length)
                                * 0.12)  # Adjust scaling factor
                    col_defs.append(
                        ColDef(col, width=width, textprops=default_textprops))

                num_rows, num_cols = df.shape

                # Dynamically adjust figsize with some padding for better scaling
                # Increased scaling factor for width and height
                figsize = (num_cols * 2, num_rows * 0.6)
                figsize = (figsize[0] + 2, figsize[1] + 2)  # Added padding

                index_name = df.columns[0]
                loop = asyncio.get_running_loop()
                fig = await loop.run_in_executor(None, stats.plot_table, df, col_defs, index_name, figsize)
                fig = fig.figure
                f = utils.plot_to_file(fig, f"plot")
                embed = discord.Embed(
                    url=url, title="Tyre Records", color=get_top_role_color(ctx.author))
                embed.set_image(url="attachment://plot.png")
                await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))
            else:
                await ctx.respond(embed=embed, view=Menu(options=headings, timeout=None), ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(description='Get F1 regulations', integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def regulations(self, ctx, year: options.SeasonOption3, type: options.RegulationOption):
        url = "https://www.fia.com/regulation/category/110"
        base_url = "https://www.fia.com"
        year = stats.roundnumber(None, year)[1]
        import requests
        from bs4 import BeautifulSoup
        html = await asyncio.to_thread(lambda: requests.get(url=url))
        s = BeautifulSoup(html.content, 'html.parser')
        results = s.find_all(class_='list-item')
        documents = [result.find('a')['href']
                     for result in results if result.find('a')]
        filtered = []

        def filter_pdfs(pdf_list, year, doc_type):

            for url in pdf_list:
                if f"_{str(year)}_" in url:
                    if doc_type.split(" ")[0].lower() in url.lower():
                        if url.startswith("https"):
                            filtered.append(url)
                        else:
                            filtered.append(base_url + url)
            return filtered
        list_pdfs = filter_pdfs(documents, year, type)
        options_list = []

        def truncate_name(name):
            name = name.split('sites/default/files/')[1]
            return name
        options_mapping = {}
        for i in list_pdfs:
            options_list.append(truncate_name(i))
            options_mapping.update({truncate_name(i): i})
        embed = discord.Embed(
            title="Choose your document:", color=get_top_role_color(ctx.author))

        class SelectMenu(discord.ui.View):

            def __init__(self, options, timeout=None):
                super().__init__(timeout=timeout)
                self.menus = self.create_select_menus(options)

                for menu in self.menus:
                    self.add_item(menu)

            def create_select_menus(self, options):
                # Split options into chunks of 25 or less
                MAX_OPTIONS = 25
                option_chunks = [options[i:i + MAX_OPTIONS]
                                 for i in range(0, len(options), MAX_OPTIONS)]
                select_menus = []

                for idx, chunk in enumerate(option_chunks):
                    select_menus.append(discord.ui.Select(
                        placeholder=f"Choose a document (Menu {idx + 1})",
                        min_values=1,
                        max_values=1,
                        options=[discord.SelectOption(
                            label=option, value=option) for option in chunk]
                    ))

                return select_menus

            # Define a callback for each select menu dynamically
            async def interaction_check(self, interaction):
                selected_value = interaction.data['values'][0]
                await self.handle_selection(interaction, selected_value)

            async def handle_selection(self, interaction, doc_name):
                await interaction.response.send_message(f"Preparing the document... (This may take a while.)", ephemeral=True)
                mypage = []
                from discord.ext.pages import Paginator, Page
                doc_name = options_mapping.get(doc_name)
                doc_response = await asyncio.to_thread(lambda: requests.get(doc_name))
                import io
                import fitz
                from PIL import Image
                pdf_stream = io.BytesIO(doc_response.content)
                doc = fitz.open(stream=pdf_stream, filetype="pdf")
                page_num = 0
                images = []
                try:
                    while True:
                        page = doc.load_page(page_num)  # number of page
                        pix = page.get_pixmap(
                            matrix=(fitz.Matrix(300 / 72, 300 / 72)))
                        img_stream = io.BytesIO()
                        img = Image.frombytes(
                            "RGB", [pix.width, pix.height], pix.samples)
                        img.save(img_stream, format="PNG")
                        # Go to the beginning of the BytesIO stream
                        img_stream.seek(0)
                        images.append(discord.File(
                            img_stream, filename=f"{page_num}.png"))
                        page_num += 1
                except ValueError:
                    pass
                doc.close()
                for idx, image in enumerate(images):
                    embed = discord.Embed(
                        title=f"Document(s) for {year} {type}",
                        color=get_top_role_color(ctx.author)
                    ).set_thumbnail(
                        url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg/1200px-F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg.png'
                    ).set_image(url=f"attachment://{idx}.png")

                    mypage.append(Page(embeds=[embed], files=[image]))

                paginator = Paginator(
                    pages=mypage, timeout=898, author_check=False)
                try:
                    await paginator.respond(ctx.interaction, ephemeral=get_ephemeral_setting(ctx))
                except discord.Forbidden:
                    return
                except discord.HTTPException:
                    return
        options = list(options_mapping.keys())
        if options == []:
            await ctx.respond("No documents found for the given combination of year and type.")
        else:
            await ctx.respond(embed=embed, view=SelectMenu(options), ephemeral=get_ephemeral_setting(ctx))


def setup(bot: discord.Bot):
    bot.add_cog(Season(bot))
