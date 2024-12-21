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
from plottable import ColDef
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
        await ctx.defer(ephemeral=get_ephemeral_setting(ctx))
        from io import StringIO
        import requests
        from bs4 import BeautifulSoup
        from plottable import ColDef
        import re

        def calculate_column_widths(df, base_width=1.8, scaling_factor=0.12):
            col_defs = []
            default_textprops = {"ha": "left"}

            for idx, col in enumerate(df.columns):
                # Max length of cell text
                max_row_length = max(df[col].apply(lambda x: len(str(x))))
                col_name_length = len(col)  # Length of the column name
                width = max(base_width, (col_name_length + max_row_length)
                            * scaling_factor)  # Calculate width
                col_defs.append(
                    ColDef(col, width=width, textprops=default_textprops))

            return col_defs

        def calculate_figsize(df, col_defs, row_scaling_factor=0.6, width_padding=2, height_padding=2):
            """Calculate figure size based on the number of columns, rows, and column widths."""
            num_rows, num_cols = df.shape
            # Sum of all column widths
            total_width = sum([col_def.width for col_def in col_defs])
            total_height = num_rows * row_scaling_factor

            # Add padding to the figure size
            figsize = (total_width + width_padding,
                       total_height + height_padding)
            return figsize

        if type == "Drivers" or type == "Sprint records" or type == "Misc. Driver records" or type == "Misc. Driver records (part 2)":
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

            def __init__(self, options, timeout=None, mapping=None):
                super().__init__(timeout=timeout)
                self.menus = self.create_select_menus(options)
                self.mapping = mapping

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
                mapping = self.mapping
                await self.handle_selection(interaction, selected_value, mapping)

            async def handle_selection(self, interaction, value, mapping):
                if type == "Races" or type == "Misc. Driver records" or type == "Misc. Driver records (part 2)":
                    record_tuple = mapping.get(value)
                    record_string = f"{record_tuple[0][1]} : {record_tuple[0][0]}"
                    embed = discord.Embed(
                        title=value, color=get_top_role_color(ctx.author))
                    embed.description = record_string
                    await interaction.edit(embed=embed)

                else:
                    for heading in headings_list:
                        text = heading.get_text(
                            strip=True).replace("[edit]", "")

                        if text.lower() == value.lower():
                            name = text
                            sibling = heading
                            table = sibling.find_next(
                                'table', class_='wikitable')
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

                    # Clean up text in the DataFrame
                    df.fillna(" ", inplace=True)
                    df.set_index(df.columns[0], inplace=True, drop=True)
                    df.drop(columns=[col for col in df.columns if col.lower(
                    ).startswith("unnamed")], inplace=True)

                    df.drop(df.index[-1], inplace=True)
                    df = df.applymap(lambda x: re.sub(
                        r'\[.*?\]', '', str(x))if isinstance(x, str) else x)
                    df = df.applymap(lambda x: re.sub(
                        r'(?<!\w)/|/(?!\w)', '', str(x)) if isinstance(x, str) else x)

                    # Calculate column definitions and dynamic figure size
                    col_defs = calculate_column_widths(df)
                    figsize = calculate_figsize(df, col_defs)

                    index_name = df.columns[0]

                    # Create the table plot asynchronously
                    loop = asyncio.get_running_loop()
                    fig = await loop.run_in_executor(None, stats.plot_table, df, col_defs, index_name, figsize)

                    fig = fig.figure
                    f = utils.plot_to_file(fig, f"plot")

                    # Send the embed with the plot
                    embed = discord.Embed(
                        title=name, color=get_top_role_color(ctx.author))
                    embed.set_image(url="attachment://plot.png")

                    await interaction.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))
        embed = discord.Embed(
            url=url, title="Choose the record to view:", color=get_top_role_color(ctx.author))
        if type == "Sprint records":
            headings = headings[119:125]
            await ctx.respond(embed=embed, view=Menu(options=headings, timeout=None, mapping=None), ephemeral=get_ephemeral_setting(ctx))
        elif type == "Drivers":
            headings = headings[:119]
            await ctx.respond(embed=embed, view=Menu(options=headings, timeout=None, mapping=None), ephemeral=get_ephemeral_setting(ctx))
        elif type == "Engines":
            headings = headings[-3:]
            await ctx.respond(embed=embed, view=Menu(options=headings, timeout=None, mapping=None), ephemeral=get_ephemeral_setting(ctx))
        elif type == "Constructors":
            await ctx.respond(embed=embed, view=Menu(options=headings, timeout=None, mapping=None), ephemeral=get_ephemeral_setting(ctx))
        elif type == "Tyres":
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
        elif type == "Races":
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
            df = df.applymap(lambda x: re.sub(
                r'\[.*?\]', '', str(x)) if isinstance(x, str) else x)
            df = df.applymap(lambda x: re.sub(
                r'(?<!\w)/|/(?!\w)', '', str(x)) if isinstance(x, str) else x)

            col_defs = []
            default_textprops = {"ha": "left"}

            df.drop(columns=[col for col in df.columns if col.lower(
            ).startswith("unnamed")], inplace=True)
            df = df.drop(columns=['Ref.'])
            options = df['Description'].tolist()
            options = list(dict.fromkeys(options))
            mapping = {
                desc: tuple(row[1:] for row in df[df['Description']
                            == desc].itertuples(index=False, name=None))
                for desc in options
            }
            embed = discord.Embed(
                url=url, title="Race Records", color=get_top_role_color(ctx.author))
            await ctx.respond(embed=embed, view=Menu(options=options, timeout=None, mapping=mapping), ephemeral=get_ephemeral_setting(ctx))

        else:
            tables = soup.find_all('table', class_='wikitable')
            table = tables[-1] if tables else None
            for span in table.find_all('span', style=lambda value: value and 'display:none' in value):
                span.decompose()
            table = str(table)
            df = pd.read_html(StringIO(str(table)))[0]
            df.fillna(" ", inplace=True)
            df = df.applymap(lambda x: re.sub(
                r'\[.*?\]', '', str(x)) if isinstance(x, str) else x)
            df = df.applymap(lambda x: re.sub(
                r'(?<!\w)/|/(?!\w)', '', str(x)) if isinstance(x, str) else x)

            # Flatten the multi-index columns
            df.columns = [' '.join(col).strip() for col in df.columns]
            df.columns = [col.replace(' Championships', '').strip()
                          for col in df.columns]

            df.drop(columns=[col for col in df.columns if col.lower(
            ).startswith("unnamed")], inplace=True)

            col_defs = []
            default_textprops = {"ha": "left"}
            df = df.drop(columns=['Ref.'])
            options = df['Description'].tolist()
            options = list(dict.fromkeys(options))
            for idx, i in enumerate(options):
                if len(i) > 100:
                    options[idx] = i[0:100]
            if type == "Misc. Driver records(part 2)":
                options = options[100:165]
            else:
                options = options[0:100]
            mapping = {
                desc[:100]: tuple(row[1:] for row in df[df['Description'] == desc].itertuples(
                    index=False, name=None))
                for desc in options
            }
            embed = discord.Embed(
                url=url, title="Misc. Driver Records", color=get_top_role_color(ctx.author))
            await ctx.respond(embed=embed, view=Menu(options=options, timeout=None, mapping=mapping), ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(description='Get F1 regulations', integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def regulations(self, ctx, year: options.SeasonOption6, type: options.RegulationOption):
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
                await interaction.response.send_message(f"Preparing the document...", ephemeral=True)
                mypage = []
                from discord.ext.pages import Paginator, Page, PaginatorButton
                doc_name = options_mapping.get(doc_name)
                doc_response = await asyncio.to_thread(lambda: requests.get(doc_name))
                import io
                import fitz
                from PIL import Image
                pdf_stream = io.BytesIO(doc_response.content)
                doc = fitz.open(stream=pdf_stream, filetype="pdf")
                page_num = 0

                class PaginatorButton(PaginatorButton):

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
                        page = doc.load_page(page_number)
                        pix = page.get_pixmap(
                            matrix=(fitz.Matrix(300 / 72, 300 / 72)))
                        img_stream = io.BytesIO()
                        img = Image.frombytes(
                            "RGB", [pix.width, pix.height], pix.samples)
                        img.save(img_stream, format="PNG")
                        img_stream.seek(0)
                        image = discord.File(
                            img_stream, filename=f"{page_number}.png")
                        embed = discord.Embed(
                            title=f"Document(s) for {year} {type}",
                            color=get_top_role_color(ctx.author)
                        ).set_thumbnail(
                            url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg/1200px-F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg.png'
                        ).set_image(url=f"attachment://{page_number}.png")
                        await interaction.edit(embed=embed, file=image)

                try:
                    while True:
                        page = doc.load_page(page_num)  # number of page
                        page_num += 1
                except ValueError:
                    pass
                page = doc.load_page(0)
                pix = page.get_pixmap(
                    matrix=(fitz.Matrix(300 / 72, 300 / 72)))
                img_stream = io.BytesIO()
                img = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples)
                img.save(img_stream, format="PNG")
                img_stream.seek(0)
                file_1 = discord.File(
                    img_stream, filename=f"0.png")
                embed_1 = discord.Embed(
                    title=f"Document(s) for {year} {type}",
                    color=get_top_role_color(ctx.author)
                ).set_thumbnail(
                    url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg/1200px-F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg.png'
                ).set_image(url=f"attachment://0.png")

                mypage.append(Page(embeds=[embed_1], files=[file_1]))
                for i in range(1, page_num):
                    embed = discord.Embed(
                        title=f"Document(s) for {year} {type}",
                        color=get_top_role_color(ctx.author)
                    ).set_thumbnail(
                        url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg/1200px-F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg.png'
                    )
                    mypage.append(Page(embeds=[embed]))
                buttons = [
                    PaginatorButton("first", label="<<",
                                    style=discord.ButtonStyle.blurple),
                    PaginatorButton("prev", label="<",
                                    style=discord.ButtonStyle.red),
                    PaginatorButton("page_indicator",
                                    style=discord.ButtonStyle.gray, disabled=True),
                    PaginatorButton("next", label=">",
                                    style=discord.ButtonStyle.green),
                    PaginatorButton("last", label=">>",
                                    style=discord.ButtonStyle.blurple),
                ]
                paginator = Paginator(
                    pages=mypage, timeout=898, author_check=False, use_default_buttons=False, custom_buttons=buttons)

                class MyModal(discord.ui.Modal):

                    def __init__(self, *args, **kwargs) -> None:
                        super().__init__(*args, **kwargs)
                        self.add_item(discord.ui.InputText(
                            label="Enter Page number:"))

                    async def callback(self, interaction: discord.Interaction):
                        page_num = int(self.children[0].value)
                        try:
                            try:
                                await paginator.goto_page(page_num-1, interaction=interaction)
                                page_number = paginator.current_page
                                page = doc.load_page(page_number)
                                pix = page.get_pixmap(
                                    matrix=(fitz.Matrix(300 / 72, 300 / 72)))
                                img_stream = io.BytesIO()
                                img = Image.frombytes(
                                    "RGB", [pix.width, pix.height], pix.samples)
                                img.save(img_stream, format="PNG")
                                img_stream.seek(0)
                                image = discord.File(
                                    img_stream, filename=f"{page_number}.png")
                                embed = discord.Embed(
                                    title=f"Document(s) for {year} {type}",
                                    color=get_top_role_color(ctx.author)
                                ).set_thumbnail(
                                    url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg/1200px-F%C3%A9d%C3%A9ration_Internationale_de_l%27Automobile_wordmark.svg.png'
                                ).set_image(url=f"attachment://{page_number}.png")
                                await interaction.edit(file=image, embed=embed)
                            except:
                                pass
                        except IndexError:
                            await interaction.response.send_message("Invalid page number.", ephemeral=True)

                class MyView(discord.ui.View):
                    @discord.ui.button(label="Switch page", row=1, style=discord.ButtonStyle.primary)
                    async def button_callback(self, button, interaction):
                        await interaction.response.send_modal(MyModal(title="Travel to your desired page."))
                try:
                    await paginator.update(interaction=ctx.interaction,
                                           custom_view=MyView())
                except:
                    pass
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
