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


API_KEY = os.getenv('WEATHER_API')

fastf1.Cache.enable_cache('cache/')
logger = logging.getLogger("f1-bot")


def schedule(ctx):
    now = pd.Timestamp.now(tz='America/New_York')

    message_embed = discord.Embed(
        title="Schedule", description="", color=get_top_role_color(ctx.author)
    )
    message_embed.set_author(name='F1 Race Schedule')

    # Fetch the current year's schedule
    schedule = fastf1.get_event_schedule(
        int(datetime.now().year), include_testing=False
    )

    # Find the index of the next event
    current_date = pd.Timestamp(date.today(), tz='UTC')
    next_event_index = schedule[schedule["Session5Date"]
                                >= current_date].index.min()

    if pd.isna(next_event_index):
        return discord.Embed(title="No upcoming events found")

    next_event = schedule.loc[next_event_index]
    race_name = next_event["EventName"]
    country = next_event["Country"]
    location = next_event["Location"]

    # Get flag emoji

    # Update embed title
    message_embed.title = f"Race Schedule for {race_name}"

    # Prepare session times dictionary
    sessions = [
        (f":one: {next_event['Session1']}", next_event["Session1Date"]),
        (f":two: {next_event['Session2']}", next_event["Session2Date"]),
        (f":three: {next_event['Session3']}", next_event["Session3Date"]),
        (f":stopwatch: {next_event['Session4']}", next_event["Session4Date"]),
        (f":checkered_flag: {next_event['Session5']}",
         next_event["Session5Date"]),
    ]

    # Convert session times to desired timezone
    sessions = {
        name: time.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M')
        for name, time in sessions if pd.notna(time)
    }

    # Add sessions to embed
    sessions_string = '\n'.join(sessions.keys())
    times_string = '\n'.join(
        f"<t:{int(pd.Timestamp(time).timestamp())}:R> <t:{int(pd.Timestamp(time).timestamp())}:F>"
        for time in sessions.values()
    )

    message_embed.add_field(name="Session", value=sessions_string, inline=True)
    message_embed.add_field(name="Time", value=times_string, inline=True)

    # Get circuit image from F1 website
    image_url = get_circuit_image(location, country)

    # Weather data (if race within 3 days)
    race_date = next_event['Session5Date'].tz_convert('America/New_York')
    time_to_race = (race_date - now).total_seconds()
    if time_to_race < 259200:  # within 3 days
        weather_data = get_weather_data(
            country, race_date.strftime('%Y-%m-%d'))
        if weather_data:
            celsius = int(weather_data['main']['temp'] - 273.15)
            fahrenheit = int(celsius * 1.8 + 32)
            message_embed.add_field(
                name='Weather Description',
                value=f"{weather_data['weather'][0]['description'].capitalize()}, "
                      f"Precipitation: {weather_data['pop']*100}%, "
                      f"Wind Speed: {weather_data['wind']['speed']}m/s",
                inline=False
            )
            message_embed.add_field(
                name='Temperature (Celsius | Fahrenheit)',
                value=f"{celsius}°C | {fahrenheit}°F", inline=False
            )
        else:
            message_embed.add_field(
                name='Weather Description', value="No weather data available", inline=False
            )

    message_embed.set_image(url=image_url.replace(" ", "%20"))
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


def get_weather_data(x, y):
    url = f'http://api.openweathermap.org/data/2.5/forecast?q={x}&appid={API_KEY}'
    response = requests.get(url)
    data = json.loads(response.content)

    # Find the weather data for the specified date
    for forecast in data['list']:
        forecast_date = datetime.fromtimestamp(
            forecast['dt']).strftime('%Y-%m-%d')
        if forecast_date == y:
            return forecast
    return None


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
        await MessageTarget(ctx).send(file=f, embed=embed)

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

        await MessageTarget(ctx).send(file=f, embed=embed)

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
        await MessageTarget(ctx).send(file=f, embed=embed)

    @commands.slash_command(description='get race schedule', integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def schedule(self, ctx):
        loop = asyncio.get_running_loop()
        message_embed = await loop.run_in_executor(None, schedule, ctx)
        await MessageTarget(ctx).send(embed=message_embed)


def setup(bot: discord.Bot):
    bot.add_cog(Season(bot))
