import asyncio
import gc
import io
import json
import logging
import math
import re
import sqlite3
import threading
import time
import urllib.parse
from datetime import date, datetime
from typing import Literal
import discord
import matplotlib.patheffects
import fastf1
from matplotlib.colors import to_rgba
import fastf1.plotting
import matplotlib
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytz
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from discord import ApplicationContext
from discord.commands import ApplicationContext
from fastf1.core import Lap, Laps, Session, SessionResults, Telemetry
from fastf1.ergast import Ergast
from fastf1.events import Event
from f1 import utils
from f1.api import ergast
from f1.errors import MissingDataError
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, MultipleLocator
from plottable import ColDef, Table
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from requests.packages.urllib3.util.retry import Retry
from unidecode import unidecode
from windrose import WindroseAxes
import wikipedia
import fitz
from PIL import Image
from f1.update import year_mapping
fastf1.plotting.setup_mpl(mpl_timedelta_support=True,
                          misc_mpl_mods=False, color_scheme='fastf1')
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"

logger = logging.getLogger("f1-bot")

ff1_erg = Ergast()
WIKI_REQUEST = 'http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='

# Disable SSL verification warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
# API Rate Limits
BURST_LIMIT = 4  # 4 requests per second
SUSTAINED_LIMIT = 500  # 500 requests per hour
# Minimum interval between requests (0.25 seconds)
REQUEST_INTERVAL = 1 / BURST_LIMIT
# Interval to avoid breaching hourly limit (7.2 seconds)
HOUR_LIMIT_INTERVAL = 3600 / SUSTAINED_LIMIT

# Global rate limiting lock
rate_lock = threading.Lock()
last_request_time = 0
request_count = 0
hour_start_time = time.time()

class customEmbed:

    # embed.set_image(url=None)
    # embed.set_footer(text='')
    # embed.description = ''
    def __init__(self, title=None, description=None, colour=None, image_url=None, thumbnail_url=None, author=None, footer=None):
        # necessary or else other info is retained in new command's embed
        self.embed = discord.Embed(title=f"Default Embed", description="")

        self.embed.clear_fields()
        self.embed.set_image(url=None)
        self.embed.set_footer(text='')
        self.embed.description = ''
        if not (title == None):
            self.embed.title = title
        if not (description == None):
            self.embed.description = description
        if (not (author == None) and len(author) == 2):
            self.embed.set_author(name=author[0], icon_url=author[1])
        if not (colour == None):
            self.embed.colour = colour
        if not (image_url == None):
            self.embed.set_image(url=image_url)
        if not (thumbnail_url == None):
            self.embed.set_thumbnail(url=thumbnail_url)
        if not (footer == None):
            self.embed.set_footer(text=footer)


class ErgastClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False

        # Retry strategy
        retries = Retry(
            total=3,  # Number of retries
            backoff_factor=0.5,  # Exponential backoff factor
            # Retry on these status codes
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def ergast_retrieve(self, api_endpoint: str):
        global last_request_time, request_count, hour_start_time

        # Rate limiting to ensure burst and sustained limits
        with rate_lock:
            current_time = time.time()

            # Check if the hourly limit is reached
            if request_count >= SUSTAINED_LIMIT:
                time_since_hour_start = current_time - hour_start_time
                if time_since_hour_start < 3600:
                    wait_time = 3600 - time_since_hour_start
                    print(
                        f"Hourly limit reached. Waiting for {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

                # Reset the counter and timestamp after waiting
                request_count = 0
                hour_start_time = time.time()

            # Ensure the interval between requests is met (burst limit)
            time_since_last_request = current_time - last_request_time
            if time_since_last_request < REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL - time_since_last_request)

            # Update last request time and increment counter
            last_request_time = time.time()
            request_count += 1

        url = f"https://api.jolpi.ca/ergast/f1/{api_endpoint}.json"

        try:
            # Reduced timeout for faster response
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            return response.json().get("MRData", None)

        except requests.exceptions.RequestException as e:
            return None


client = ErgastClient()

def convert_shootout_to_qualifying(year, session):
    if session == "Sprint Qualifying" and year == 2023:
        session = "Sprint Shootout"
        return session
    else:
        return session


def get_random_color():
    used_colors = set()
    import random
    while True:
        # Generate a random color hex code
        color = f'#{random.randint(0, 0xFFFFFF):06x}'
        if color not in used_colors:
            used_colors.add(color)
            return color


def lap_filter_sc(row: pd.Series) -> bool:
    return "4" in row.loc["TrackStatus"]


def lap_filter_vsc(row: pd.Series) -> bool:
    return (("6" in row.loc["TrackStatus"]) or ("7" in row.loc["TrackStatus"])) and (
        "4" not in row.loc["TrackStatus"]
    )


def lap_filter_red_flag(row: pd.Series) -> bool:
    return (("5" in row.loc["TrackStatus"]))


def find_sc_laps(df_laps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sc_laps = np.sort(df_laps[df_laps.apply(
        lap_filter_sc, axis=1)]["LapNumber"].unique())
    vsc_laps = np.sort(df_laps[df_laps.apply(
        lap_filter_vsc, axis=1)]["LapNumber"].unique())
    return sc_laps, vsc_laps


def find_red_laps(df_laps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    red_laps = np.sort(df_laps[df_laps.apply(
        lap_filter_red_flag, axis=1)]["LapNumber"].unique())
    return red_laps


def shade_sc_periods(sc_laps: np.ndarray, vsc_laps: np.ndarray, ax):
    sc_laps = np.append(sc_laps, [-1])
    vsc_laps = np.append(vsc_laps, [-1])

    def plot_periods(laps, label, color="orange"):
        start = 0
        end = 1

        while end < len(laps):
            # Check if the current SC period is still ongoing
            if laps[end] == laps[end - 1] + 1:
                end += 1
            else:
                if end - start > 1:
                    # SC period lasts for more than one lap
                    ax.axvspan(
                        xmin=laps[start] - 1,
                        xmax=laps[end - 1] - 1,
                        alpha=0.5,
                        color=color,
                        label=label if start == 0 else "_"

                    )
                else:
                    # SC period lasts only one lap
                    ax.axvspan(
                        xmin=laps[start] - 1,
                        xmax=laps[start],
                        alpha=0.5,
                        color=color,
                        label=label if start == 0 else "_"

                    )
                start = end
                end += 1

    plot_periods(sc_laps, "SC")
    plot_periods(vsc_laps, "VSC", "yellow")


def shade_red_flag(red_laps: np.ndarray, ax):
    red_laps = np.append(red_laps, [-1])

    def plot_periods(laps, label, hatch=None):
        start = 0
        end = 1

        while end < len(laps):
            if laps[end] == laps[end - 1] + 1:
                end += 1
            else:
                if end - start > 1:
                    ax.axvspan(
                        xmin=laps[start] - 1,
                        xmax=laps[end - 1] - 1,
                        alpha=0.5,
                        color="red",
                        label=label if start == 0 else "_",
                        hatch=hatch,
                    )
                else:
                    ax.axvspan(
                        xmin=laps[start] - 1,
                        xmax=laps[start],
                        alpha=0.5,
                        color="red",
                        label=label if start == 0 else "_",
                        hatch=hatch,
                    )
                start = end
                end += 1

    plot_periods(red_laps, "Red Flag")


def get_event_note(year, eventname):
    base_url = "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-"
    url = base_url + year_mapping.get(year)
    if eventname == '70th Anniversary Grand Prix':
        eventname = 'Formula 1 70th Anniversary Grand Prix'
    elif eventname == 'Mexico City Grand Prix' and year < 2023:
        eventname = 'Mexican Grand Prix'
    elif eventname == 'São Paulo Grand Prix' and year < 2023:
        eventname = 'Brazilian Grand Prix'
    elif eventname == 'Saudi Arabian Grand Prix' and year < 2023:
        eventname = 'Saudi Arabia Grand Prix'
    url = url + f"/event/{eventname.replace(' ', '%20')}"
    resp = requests.get(url)
    docs = re.findall(r'href="(.+?).pdf"', resp.text)
    docs = [doc for doc in docs if re.match(r'.*?((event-?_? ?notes)|(pirelli)).*?', doc, re.IGNORECASE)]
    return docs

def get_pdf(url: str):
    resp = requests.get(url)
    cnt = 0
    while not resp.ok and cnt < 3:
        resp = requests.get(url)
        cnt += 1
    return resp.content

def parse_event_pdf(pdf: bytes):
    pdf = fitz.open(stream=pdf, filetype="pdf")
    for page in pdf:
        text = page.get_text()
        if 'Compound' in text:
            compound = set(re.findall(r'(?=\D(C\d)\D)', text))
            return list(compound)
        
async def get_pdf_async(session, url: str):
    async with session.get(url) as response:
        return await response.read()

async def get_compound_async(year: int, eventname: str):
    import aiohttp
    docs = get_event_note(year, eventname)

    # Use async to fetch PDFs concurrently
    urls = [f'https://www.fia.com{doc}.pdf' for doc in docs]
    async with aiohttp.ClientSession() as session:
        pdf_contents = await asyncio.gather(*[get_pdf_async(session, url) for url in urls])

    # Parse PDFs and extract compounds
    compounds = []
    for pdf in pdf_contents:
        compound = parse_event_pdf(pdf)
        if compound:
            compounds.extend(compound)

    return compounds if compounds else None


async def sectors_func(yr, rc, sn, d1, d2, lap, event, session):
    d1 = d1[0:3].upper()
    d2 = d2[0:3].upper()

    lap1 = lap
    lap2 = lap1

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')

    # Explore the lap data
    await asyncio.to_thread(lambda: session.laps)
    circuit_info = session.get_circuit_info()
    track_angle = circuit_info.rotation / 180 * np.pi

    def rotate(xy, *, angle):
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)]])
        return np.matmul(xy, rot_mat)

    if (d1 == None or d1 == ''):
        d1 = session.laps.pick_fastest()['Driver']

    if (d2 == None or d2 == ''):
        d2 = session.laps.pick_fastest()['Driver']

    driver_1 = d1
    driver_2 = d2

    color_1 = fastf1.plotting.get_driver_color(driver_1, session=session)
    color_2 = fastf1.plotting.get_driver_color(driver_2, session=session)
    # Find the laps
    laps_driver_1 = session.laps.pick_drivers(driver_1)
    laps_driver_2 = session.laps.pick_drivers(driver_2)

    if (lap1 == None or lap1 == ''):
        fastest_driver_1 = laps_driver_1.pick_fastest()
    else:
        fastest_driver_1 = laps_driver_1[laps_driver_1['LapNumber'] == int(
            lap1)].iloc[0]

    if (lap2 == None or lap2 == ''):
        fastest_driver_2 = laps_driver_2.pick_fastest()
    else:
        fastest_driver_2 = laps_driver_2[laps_driver_2['LapNumber'] == int(
            lap2)].iloc[0]

    telemetry_driver_1 = await asyncio.to_thread(lambda: fastest_driver_1.get_telemetry())
    telemetry_driver_2 = await asyncio.to_thread(lambda: fastest_driver_2.get_telemetry())

    # Identify team colors
    team_driver_1 = laps_driver_1['Team'].iloc[0]
    team_driver_2 = laps_driver_2['Team'].iloc[0]

    # Merge the telemetry from both drivers into one dataframe
    telemetry_driver_1['Driver'] = driver_1
    telemetry_driver_2['Driver'] = driver_2

    telemetry = pd.concat([telemetry_driver_1, telemetry_driver_2])

    # Calculate minisectors
    num_minisectors = 80
    total_distance = max(telemetry['Distance'])
    minisector_length = total_distance / num_minisectors

    minisectors = [0]

    for i in range(0, (num_minisectors - 1)):
        minisectors.append(minisector_length * (i + 1))

    # Assign a minisector number to every row in the telemetry dataframe
    telemetry['Minisector'] = telemetry['Distance'].apply(
        lambda dist: (
            int((dist // minisector_length) + 1)
        )
    )

    # Calculate minisector speeds per driver
    average_speed = telemetry.groupby(['Minisector', 'Driver'])[
        'Speed'].mean().reset_index()

    # Per minisector, find the fastest driver
    fastest_driver = average_speed.loc[average_speed.groupby(['Minisector'])[
        'Speed'].idxmax()]
    fastest_driver = fastest_driver[['Minisector', 'Driver']].rename(
        columns={'Driver': 'Fastest_driver'})

    # Merge the fastest_driver dataframe to the telemetry dataframe on minisector
    telemetry = telemetry.merge(fastest_driver, on=['Minisector'])
    telemetry = telemetry.sort_values(by=['Distance'])

    # Since our plot can only work with integers, we need to convert the driver abbreviations to integers (1 or 2)
    telemetry.loc[telemetry['Fastest_driver']
                  == driver_1, 'Fastest_driver_int'] = 1
    telemetry.loc[telemetry['Fastest_driver']
                  == driver_2, 'Fastest_driver_int'] = 2

    # Get the x and y coordinates
    x = np.array(telemetry['X'].values)
    y = np.array(telemetry['Y'].values)

    coords = np.column_stack((x, y))  # Combine x and y into a single array
    rotated_coords = rotate(coords, angle=track_angle)
    x_rotated, y_rotated = rotated_coords[:, 0], rotated_coords[:, 1]
    points = np.array([x_rotated, y_rotated]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fastest_driver_array = telemetry['Fastest_driver_int'].to_numpy().astype(
        float)

    # The segments we just created can now be colored according to the fastest driver in a minisector
    cmap = ListedColormap([color_1, color_2])
    rgba_colors = [to_rgba(color) for color in cmap.colors]

# Check for identical colors and adjust alpha
    adjusted_colors = []
    for i, color in enumerate(rgba_colors):
        # Compare RGB (ignore alpha for comparison)
        if i > 0 and color[:3] == rgba_colors[i - 1][:3]:
            # Reduce alpha if the colors are the same
            # Reduce alpha by 0.5, ensure it's not negative
            new_color = (*color[:3], max(0, color[3] - 0.9))
        else:
            new_color = color
        adjusted_colors.append(new_color)

    # Create a new colormap with adjusted colors
    cmap = ListedColormap(adjusted_colors)
    lc_comp = LineCollection(
        segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
    lc_comp.set_array(fastest_driver_array)
    lc_comp.set_linewidth(5)
    # Create the plot
    plt.rcParams['figure.figsize'] = [18, 10]
    plt.rcParams["figure.autolayout"] = True

    # Plot the line collection and style the plot
    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False,
                    labelbottom=False, bottom=False)

    # Add a colorbar for as legend
    cbar = plt.colorbar(mappable=lc_comp, boundaries=np.arange(1, 4))
    cbar.set_ticks(np.arange(1.5, 3.5))
    cbar.set_ticklabels([driver_1, driver_2])

    if (lap1 == None or lap1 == ''):
        lap1 = "Fastest Lap"
    else:
        lap1 = "Lap " + str(lap1)
    if (lap2 == None or lap2 == ''):
        lap2 = "Fastest Lap"
    else:
        lap2 = "Lap " + str(lap2)

    plt.suptitle(f"{yr} {event['EventName']} {sn} - Fastest Sectors\n" +
                 d1 + " (" + f"{utils.format_timedelta(fastest_driver_1['LapTime'])}"+ ") vs " + d2 + " (" + f"{utils.format_timedelta(fastest_driver_2['LapTime'])}" + ")")
    file = utils.plot_to_file(plt.gcf(), "plot")
    return file


async def weather(year, location, session, event, race):

    race_name = race.event.EventName
    df = await asyncio.to_thread(lambda: race.laps)

# load dataframe of df (by Final Position in ascending order)
    df = df.sort_values(by=['LapNumber', 'Position'], ascending=[
                        False, True]).reset_index(drop=True)

# fill in empty laptime records and convert to seconds
    df.LapTime = df.LapTime.fillna(
        df['Sector1Time']+df['Sector2Time']+df['Sector3Time'])
    df.LapTime = df.LapTime.dt.total_seconds()
    df.Sector1Time = df.Sector1Time.dt.total_seconds()
    df.Sector2Time = df.Sector2Time.dt.total_seconds()
    df.Sector3Time = df.Sector3Time.dt.total_seconds()

# weather
    df_weather = race.weather_data.copy()
    df_weather['Time'] = df_weather['Time'].dt.total_seconds()/60
    df_weather = df_weather.rename(columns={'Time': 'SessionTime(Minutes)'})

# Rain Indicator
    rain = df_weather.Rainfall.eq(True).any()


# get session

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Weather Data & Track Evolution \n'+race_name)

# Track and Air Temperature
    sns.lineplot(data=df_weather, x='SessionTime(Minutes)',
                 y='TrackTemp', label='TrackTemp', ax=ax[0, 0])
    sns.lineplot(data=df_weather, x='SessionTime(Minutes)',
                 y='AirTemp', label='AirTemp', ax=ax[0, 0])
    if rain:
        ax[0, 0].fill_between(df_weather[df_weather.Rainfall == True]['SessionTime(Minutes)'], df_weather.TrackTemp.max(
        )+0.5, df_weather.AirTemp.min()-0.5, facecolor="blue", color='blue', alpha=0.1, zorder=0, label='Rain')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].set_ylabel('Temperature')
    ax[0, 0].title.set_text('Track Temperature & Air Temperature (°C)')

# Humidity
    sns.lineplot(df_weather, x='SessionTime(Minutes)',
                 y='Humidity', ax=ax[0, 1])
    if rain:
        ax[0, 1].fill_between(df_weather[df_weather.Rainfall == True]['SessionTime(Minutes)'], df_weather.Humidity.max(
        )+0.5, df_weather.Humidity.min()-0.5, facecolor="blue", color='blue', alpha=0.1, zorder=0, label='Rain')
        ax[0, 1].legend(loc='upper right')
    ax[0, 1].title.set_text('Track Humidity (%)')

# Pressure
    sns.lineplot(data=df_weather, x='SessionTime(Minutes)',
                 y='Pressure', ax=ax[1, 0])
    ax[1, 0].title.set_text('Air Pressure (mbar)')

    # Wind Direction & Speed
    rect = ax[1, 1].get_position()
    wax = WindroseAxes(fig, rect)
    fig.add_axes(wax)
    wax.bar(df_weather.WindDirection, df_weather.WindSpeed,
            normed=True, opening=0.8, edgecolor='white')
    wax.set_legend()
    ax[1, 1].title.set_text('Wind Direction (°) and Speed(m/s)')
    fig.set_tight_layout(False)
    ax[0, 0].grid(which="minor", alpha=0.1)
    ax[0, 0].minorticks_on()
    ax[0, 1].grid(which="minor", alpha=0.1)
    ax[0, 1].minorticks_on()
    ax[1, 0].grid(which="minor", alpha=0.1)
    ax[1, 0].minorticks_on()
    ax[1, 1].grid(which="minor", alpha=0.1)
    ax[1, 1].minorticks_on()

    file = utils.plot_to_file(fig, "plot")
    return file


async def cornering_func(yr, rc, sn, d1, d2, lap1, lap2, dist1, dist2, event, session):

    d1 = d1[0:3].upper()
    d2 = d2[0:3].upper()

    # Get the laps
    laps = await asyncio.to_thread(lambda: session.laps)

    if (d1 == None or d1 == ''):
        d1 = laps.pick_fastest()['Driver']

    if (d2 == None or d2 == ''):
        d2 = laps.pick_fastest()['Driver']

    # Setting parameters
    driver_1, driver_2 = d1, d2

    car_data = await asyncio.to_thread(lambda: laps.pick_drivers(
        driver_1).pick_fastest().get_car_data().add_distance())
    dist = car_data['Distance']
    maxdist = dist[len(dist)-1]

    if (dist1 == None or dist1 == ''):
        dist1 = 0

    if (dist2 == None or dist2 == ''):
        dist2 = maxdist

    if (dist1 > dist2):
        dist1, dist2 = dist2, dist1

    distance_min, distance_max = dist1, dist2

    # Extracting the laps
    laps_driver_1 = await asyncio.to_thread(lambda: laps.pick_drivers(driver_1))
    laps_driver_2 = await asyncio.to_thread(lambda: laps.pick_drivers(driver_2))

    if (lap1 == None or lap1 == ''):
        telemetry_driver_1 = await asyncio.to_thread(lambda: laps_driver_1.pick_fastest().get_car_data().add_distance())
        driver1_laptime =  laps_driver_1.pick_fastest()['LapTime']
    else:
        temp_laps1 = laps_driver_1[laps_driver_1['LapNumber'] == int(
            lap1)].iloc[0]
        driver1_laptime =  temp_laps1['LapTime']
        telemetry_driver_1 = await asyncio.to_thread(lambda: temp_laps1.get_car_data().add_distance())

    if (lap2 == None or lap2 == ''):
        telemetry_driver_2 = await asyncio.to_thread(lambda: laps_driver_2.pick_fastest().get_car_data().add_distance())
        driver2_laptime =  laps_driver_2.pick_fastest()['LapTime']
    else:
        temp_laps2 = laps_driver_2[laps_driver_2['LapNumber'] == int(
            lap2)].iloc[0]
        driver2_laptime =  temp_laps2['LapTime']
        telemetry_driver_2 = await asyncio.to_thread(lambda: temp_laps2.get_car_data().add_distance())

    # Identifying the team for coloring later on
    team_driver_1 = laps_driver_1.reset_index().loc[0, 'Team']
    team_driver_2 = laps_driver_2.reset_index().loc[0, 'Team']

    # Assigning labels to what the drivers are currently doing
    telemetry_driver_1.loc[telemetry_driver_1['Brake']
                           > 0, 'CurrentAction'] = 'Brake'
    telemetry_driver_1.loc[telemetry_driver_1['Throttle']
                           == 100, 'CurrentAction'] = 'Full Throttle'
    telemetry_driver_1.loc[(telemetry_driver_1['Brake'] == 0) & (
        telemetry_driver_1['Throttle'] < 100), 'CurrentAction'] = 'Cornering'

    telemetry_driver_2.loc[telemetry_driver_2['Brake']
                           > 0, 'CurrentAction'] = 'Brake'
    telemetry_driver_2.loc[telemetry_driver_2['Throttle']
                           == 100, 'CurrentAction'] = 'Full Throttle'
    telemetry_driver_2.loc[(telemetry_driver_2['Brake'] == 0) & (
        telemetry_driver_2['Throttle'] < 100), 'CurrentAction'] = 'Cornering'

    # Numbering each unique action to identify changes, so that we can group later on
    telemetry_driver_1['ActionID'] = (
        telemetry_driver_1['CurrentAction'] != telemetry_driver_1['CurrentAction'].shift(1)).cumsum()
    telemetry_driver_2['ActionID'] = (
        telemetry_driver_2['CurrentAction'] != telemetry_driver_2['CurrentAction'].shift(1)).cumsum()

    # Identifying all unique actions
    actions_driver_1 = telemetry_driver_1[['ActionID', 'CurrentAction', 'Distance']].groupby(
        ['ActionID', 'CurrentAction']).max('Distance').reset_index()
    actions_driver_2 = telemetry_driver_2[['ActionID', 'CurrentAction', 'Distance']].groupby(
        ['ActionID', 'CurrentAction']).max('Distance').reset_index()

    actions_driver_1['Driver'] = driver_1
    actions_driver_2['Driver'] = driver_2

    # Calculating the distance between each action, so that we know how long the bar should be
    actions_driver_1['DistanceDelta'] = actions_driver_1['Distance'] - \
        actions_driver_1['Distance'].shift(1)
    actions_driver_1.loc[0,
                         'DistanceDelta'] = actions_driver_1.loc[0, 'Distance']

    actions_driver_2['DistanceDelta'] = actions_driver_2['Distance'] - \
        actions_driver_2['Distance'].shift(1)
    actions_driver_2.loc[0,
                         'DistanceDelta'] = actions_driver_2.loc[0, 'Distance']

    # Merging together
    all_actions = pd.concat([actions_driver_1, actions_driver_2])

    # Calculating average speed
    avg_speed_driver_1 = np.mean(telemetry_driver_1['Speed'].loc[
        (telemetry_driver_1['Distance'] >= distance_min) &
        (telemetry_driver_1['Distance'] <= distance_max) &
        ~telemetry_driver_1['Speed'].isna()  # Exclude NaN values
    ])
    avg_speed_driver_2 = np.mean(telemetry_driver_2['Speed'].loc[
        (telemetry_driver_2['Distance'] >= distance_min) &
        (telemetry_driver_2['Distance'] <= distance_max) &
        ~telemetry_driver_2['Speed'].isna()  # Exclude NaN values
    ])

    if avg_speed_driver_1 > avg_speed_driver_2:
        speed_text = f"{driver_1} {round(avg_speed_driver_1 - avg_speed_driver_2,2)} km/h faster"
    else:
        speed_text = f"{driver_2} {round(avg_speed_driver_2 - avg_speed_driver_1,2)} km/h faster"

    plt.rcParams["figure.figsize"] = [13, 4]
    plt.rcParams["figure.autolayout"] = True

    telemetry_colors = {
        'Full Throttle': 'green',
        'Cornering': 'grey',
        'Brake': 'red',
    }

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
    fig, ax = plt.subplots(2)

    style1 = fastf1.plotting.get_driver_style(identifier=driver_1,
                                              style=['color', 'linestyle'],
                                              session=session)
    style2 = fastf1.plotting.get_driver_style(identifier=driver_2,
                                              style=['color', 'linestyle'],
                                              session=session)

    try:
        ax[0].plot(telemetry_driver_1['Distance'], telemetry_driver_1['Speed'], **style1,
                   label=driver_1)
    except:
        ax[0].plot(telemetry_driver_1['Distance'],
                   telemetry_driver_1['Speed'], label=driver_1, color='grey')

    try:
        if (d1 != d2):
            ax[0].plot(telemetry_driver_2['Distance'], telemetry_driver_2['Speed'], **style2,
                       label=driver_2)
        else:
            ax[0].plot(telemetry_driver_2['Distance'],
                       telemetry_driver_2['Speed'], label=driver_2, color='#777777')
    except:
        ax[0].plot(telemetry_driver_2['Distance'],
                   telemetry_driver_2['Speed'], label=driver_2, color='grey')

    # Speed difference
    if distance_min == None:
        ax[0].text(0, 200, speed_text, fontsize=15)
    else:
        ax[0].text(distance_min + 15, 200, speed_text, fontsize=15)

    ax[0].set(ylabel='Speed in km/h')
    ax[0].legend(loc="lower right")

    for driver in [driver_1, driver_2]:
        driver_actions = all_actions.loc[all_actions['Driver'] == driver]

        previous_action_end = 0
        for _, action in driver_actions.iterrows():
            ax[1].barh(
                [driver],
                action['DistanceDelta'],
                left=previous_action_end,
                color=telemetry_colors[action['CurrentAction']]
            )

            previous_action_end = previous_action_end + action['DistanceDelta']

    plt.xlabel('Track Distance in meters')

    # Invert y-axis
    plt.gca().invert_yaxis()

    # Remove frame from plot
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)

    # Add legend
    labels = list(telemetry_colors.keys())
    handles = [plt.Rectangle(
        (0, 0), 1, 1, color=telemetry_colors[label]) for label in labels]
    ax[1].legend(handles, labels)

    # Zoom in on the specific part we want to see
    ax[0].set_xlim(distance_min, distance_max)
    ax[1].set_xlim(distance_min, distance_max)

    if (lap1 == None or lap1 == ''):
        lap1 = "Fastest Lap"
    else:
        lap1 = "Lap " + str(lap1)
    if (lap2 == None or lap2 == ''):
        lap2 = "Fastest Lap"
    else:
        lap2 = "Lap " + str(lap2)

    ax[0].grid(which="minor", alpha=0.1)
    ax[0].minorticks_on()

    plt.suptitle(f"{yr} {event['EventName']} {sn}\n" +
                 d1 + " (" + f"{utils.format_timedelta(driver1_laptime)}" + ") vs " + d2 + " (" + f"{utils.format_timedelta(driver2_laptime)}"  + ")",)
    file = utils.plot_to_file(fig, "plot")
    return file


def heatmap_func(yr):
    ergast = Ergast()
    races = ergast.get_race_schedule(yr)
    results = []
    if yr == int(datetime.now().year):
        schedule = fastf1.get_event_schedule(yr, include_testing=False)
        last_index = None
        for index, row in schedule.iterrows():
            if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                last_index = index
        try:
            for rnd, race in races['raceName'].items():
                if rnd >= last_index:
                    break
                temp = ergast.get_race_results(season=yr, round=rnd + 1)
                temp = temp.content[0]

        # If there is a sprint, get the results as well
                sprint = ergast.get_sprint_results(season=yr, round=rnd + 1)
                if sprint.content and sprint.description['round'][0] == rnd + 1:
                    temp = pd.merge(
                        temp, sprint.content[0], on='driverCode', how='left')
            # Add sprint points and race points to get the total
                    temp['points'] = temp['points_x'] + temp['points_y']
                    temp.drop(columns=['points_x', 'points_y'], inplace=True)

        # Add round no. and grand prix name
                temp['round'] = rnd + 1
                temp['race'] = race.removesuffix(' Grand Prix')
                # Keep useful cols.
                temp = temp[['round', 'race', 'driverCode', 'points']]
                results.append(temp)

        except IndexError:
            results = []
            for rnd, race in races['raceName'].items():
                if rnd >= last_index-1:
                    break
                temp = ergast.get_race_results(season=yr, round=rnd + 1)
                temp = temp.content[0]

        # If there is a sprint, get the results as well
                sprint = ergast.get_sprint_results(season=yr, round=rnd + 1)
                if sprint.content and sprint.description['round'][0] == rnd + 1:
                    temp = pd.merge(
                        temp, sprint.content[0], on='driverCode', how='left')
            # Add sprint points and race points to get the total
                    temp['points'] = temp['points_x'] + temp['points_y']
                    temp.drop(columns=['points_x', 'points_y'], inplace=True)

        # Add round no. and grand prix name
                temp['round'] = rnd + 1
                temp['race'] = race.removesuffix(' Grand Prix')
                # Keep useful cols.
                temp = temp[['round', 'race', 'driverCode', 'points']]
                results.append(temp)

    else:
        # For each race in the season
        for rnd, race in races['raceName'].items():

            # Get results. Note that we use the round no. + 1, because the round no.
            # starts from one (1) instead of zero (0)
            temp = ergast.get_race_results(season=yr, round=rnd + 1)
            temp = temp.content[0]

    # If there is a sprint, get the results as well
            sprint = ergast.get_sprint_results(season=yr, round=rnd + 1)
            if sprint.content and sprint.description['round'][0] == rnd + 1:
                temp = pd.merge(
                    temp, sprint.content[0], on='driverCode', how='left')
        # Add sprint points and race points to get the total
                temp['points'] = temp['points_x'] + temp['points_y']
                temp.drop(columns=['points_x', 'points_y'], inplace=True)

    # Add round no. and grand prix name
            temp['round'] = rnd + 1
            temp['race'] = race.removesuffix(' Grand Prix')
            # Keep useful cols.
            temp = temp[['round', 'race', 'driverCode', 'points']]
            results.append(temp)

# Append all races into a single dataframe
    results = pd.concat(results)
    races = results['race'].drop_duplicates()
    results = results.pivot(
        index='driverCode', columns='round', values='points')
# Here we have a 22-by-22 matrix (22 races and 22 drivers, incl. DEV and HUL)

# Rank the drivers by their total points
    results['total_points'] = results.sum(axis=1)
    results = results.sort_values(by='total_points', ascending=False)
    results.drop(columns='total_points', inplace=True)

# Use race name, instead of round no., as column names
    results.columns = races
    fig = px.imshow(
        results,
        text_auto=True,
        aspect='auto',  # Automatically adjust the aspect ratio
        color_continuous_scale=[[0,    'rgb(198, 219, 239)'],  # Blue scale
                                [0.25, 'rgb(107, 174, 214)'],
                                [0.5,  'rgb(33,  113, 181)'],
                                [0.75, 'rgb(8,   81,  156)'],
                                [1,    'rgb(8,   48,  107)']],
        labels={'x': 'Race',
                'y': 'Driver',
                'color': 'Points'}       # Change hover texts
    )
    fig.update_xaxes(title_text='')      # Remove axis titles
    fig.update_yaxes(title_text='')
    fig.update_yaxes(tickmode='linear')  # Show all ticks, i.e. driver names
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey',
                     showline=False,
                     tickson='boundaries')              # Show horizontal grid only
    # And remove vertical grid
    fig.update_xaxes(showgrid=False, showline=False)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')     # White background
    fig.update_layout(coloraxis_showscale=False)        # Remove legend
    fig.update_layout(xaxis=dict(side='top'))           # x-axis on top
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig_bytes = fig.to_image(format="png")
    buffer = io.BytesIO()
    buffer.write(fig_bytes)
    buffer.seek(0)
    file = discord.File(buffer, filename="plot.png")
    buffer.close()
    plt.close()

    return file


async def tel_func(yr, rc, sn, d1, d2, lap1, lap2, event, session):
    d1 = d1[0:3].upper()
    d2 = d2[0:3].upper()

    laps = session.laps

    if (d1 == None or d1 == ''):
        d1 = await asyncio.to_thread(lambda: laps.pick_fastest()['Driver'])

    if (d2 == None or d2 == ''):
        d2 = await asyncio.to_thread(lambda: laps.pick_fastest()['Driver'])

    drv1 = d1
    drv2 = d2

    first_driver = laps.pick_drivers(drv1)
    first_driver_info = session.get_driver(drv1)
    my_styles = [
        {'linestyle': 'solid', 'color': 'auto', 'custom_arg': True},
        {'linestyle': 'dashed', 'color': 'auto', 'other_arg': 10}
    ]

    style1 = fastf1.plotting.get_driver_style(identifier=d1,
                                              style=['color', 'linestyle'],
                                              session=session)
    style2 = fastf1.plotting.get_driver_style(identifier=d2,
                                              style=['color', 'linestyle'],
                                              session=session)

    second_driver = laps.pick_drivers(drv2)
    second_driver_info = session.get_driver(drv2)

    if (lap1 == None or lap1 == ''):
        first_driver = laps.pick_drivers(drv1).pick_fastest()
    else:
        driver_laps = session.laps.pick_drivers(drv1)
        first_driver = driver_laps[driver_laps['LapNumber'] == int(
            lap1)].iloc[0]

    if (lap2 == None or lap2 == ''):
        second_driver = await asyncio.to_thread(lambda: laps.pick_drivers(drv2).pick_fastest())
    else:
        driver_laps = await asyncio.to_thread(lambda: session.laps.pick_drivers(drv2))
        second_driver = driver_laps[driver_laps['LapNumber'] == int(
            lap2)].iloc[0]
    first_car = await asyncio.to_thread(lambda: first_driver.get_car_data().add_distance())
    second_car = await asyncio.to_thread(lambda: second_driver.get_car_data().add_distance())

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
    fig, ax = plt.subplots(7, 1, figsize=(15, 12), dpi=300, gridspec_kw={
        # Equal height for all subplots
        'height_ratios': [3, 3, 3, 3, 3, 3, 3],
        'hspace': 0.5  # Adjust spacing between subplots
    })

    if (lap1 == None or lap1 == ''):
        lap1 = "Fastest Lap"
    else:
        lap1 = "Lap " + str(lap1)
    if (lap2 == None or lap2 == ''):
        lap2 = "Fastest Lap"
    else:
        lap2 = "Lap " + str(lap2)

    fig.suptitle(f"{yr} {event['EventName']} {sn}\n" +
                 drv1 + " (" + utils.format_timedelta(first_driver['LapTime']) + ") vs " + drv2 + " (" + utils.format_timedelta(second_driver['LapTime']) + ")")

    drs_1 = first_car['DRS']
    drs_2 = second_car['DRS']

    brake_2 = second_car['Brake']

    drs1 = []
    drs2 = []

    d = 0
    while (d < len(drs_1)):
        if (drs_1[d] >= 10 and drs_1[d] % 2 == 0):
            drs1.extend([1])
        else:
            drs1.extend([0])
        d += 1
    d = 0
    while (d < len(drs_2)):
        if (drs_2[d] >= 10 and drs_2[d] % 2 == 0):
            drs2.extend([-1])
        else:
            drs2.extend([0])
        d += 1

    brake2 = []

    b = 0
    while (b < len(brake_2)):
        if (brake_2[b] == 1):
            brake2.extend([-1])
        else:
            brake2.extend([0])
        b += 1
    if (len(brake_2) < len(second_car['Distance'])):
        b = len(brake_2)
        while (b < len(second_car['Distance'])):
            brake_2.extend([0])
            b += 1
    delta_time, ref_tel, compare_tel = fastf1.utils.delta_time(
        first_driver, second_driver)

    delta = []

    dt = 0
    while (dt < len(first_car['Distance'])):
        delta.extend([float(delta_time[dt])*(-1)])
        dt += 1

    ax[6].set_ylabel(f"Delta (s)\n {drv1} | {drv2}")

    l2, = ax[0].plot(second_car['Distance'],
                     second_car['Speed'], **style2)
    l1, = ax[0].plot(first_car['Distance'],
                     first_car['Speed'], **style1)
    ax[1].plot(second_car['Distance'], second_car['RPM'], **style2)
    ax[1].plot(first_car['Distance'], first_car['RPM'], **style1)
    ax[2].plot(second_car['Distance'], second_car['nGear'], **style2)
    ax[2].plot(first_car['Distance'], first_car['nGear'], **style1)
    ax[3].plot(second_car['Distance'],
               second_car['Throttle'], **style2)
    ax[3].plot(first_car['Distance'], first_car['Throttle'], **style1)
    ax[6].plot(first_car['Distance'], delta, color='white')

    ax[0].set_ylabel("Speed (km/h)")
    ax[1].set_ylabel("RPM (#)")
    ax[2].set_ylabel("Gear (#)")
    ax[3].set_ylabel("Throttle (%)")
    ax[4].set_ylabel(f"Brake (%)\n {drv2} | {drv1}")
    ax[5].set_ylabel("DRS (Binary)")

    fig.align_ylabels()
    fig.legend((l1, l2), (drv1, drv2))

    ax[5].fill_between(second_car['Distance'], drs2,
                       step="pre", **style2, alpha=0.5)
    ax[5].fill_between(first_car['Distance'], drs1,
                       step="pre", **style1, alpha=1)
    ax[4].fill_between(second_car['Distance'], brake2,
                       step="pre", **style2, alpha=0.5)
    ax[4].fill_between(first_car['Distance'], first_car['Brake'],
                       step="pre", **style1, alpha=1)

    plt.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.05)

    ax[2].get_yaxis().set_major_locator(MaxNLocator(integer=True))

    ticks = ax[6].get_yticks()
    # set labels to absolute values and with integer representation
    ax[6].set_yticklabels([round(abs(tick), 1) for tick in ticks])

    ax[0].grid(which="minor", alpha=0.1)
    ax[0].minorticks_on()
    ax[1].grid(which="minor", alpha=0.1)
    ax[1].minorticks_on()
    ax[2].grid(which="minor", alpha=0.1)
    ax[2].minorticks_on()
    ax[3].grid(which="minor", alpha=0.1)
    ax[3].minorticks_on()
    ax[4].grid(which="minor", alpha=0.1)
    ax[4].minorticks_on()
    ax[5].grid(which="minor", alpha=0.1)
    ax[5].minorticks_on()
    ax[6].grid(which="minor", alpha=0.1)
    ax[6].minorticks_on()
    ax[0].set_xlabel("Track Distance (m)")
    ax[1].set_xlabel("Track Distance (m)")
    ax[2].set_xlabel("Track Distance (m)")
    ax[3].set_xlabel("Track Distance (m)")
    ax[4].set_xlabel("Track Distance (m)")
    ax[5].set_xlabel("Track Distance (m)")
    ax[6].set_xlabel("Track Distance (m)")
    plt.tight_layout()
    file = utils.plot_to_file(fig, "plot")
    return file


def driver_func(yr):

    import datetime
    schedule = fastf1.get_event_schedule(yr, include_testing=False)
    if yr == int(datetime.datetime.now().year):

        last_index = None
        for index, row in schedule.iterrows():

            if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                number = row['RoundNumber']
                last_index = number

        last_round = last_index
    else:
        last_round = max(schedule.RoundNumber)

    colors = ["#333333", "#444444", "#555555", "#666666", "#777777",
              "#888888", "#999999", "#AAAAAA", "#BBBBBB", "#CCCCCC"]
    color_counter = 0

    # Specify the number of rounds we want in our plot (in other words, specify the current round)
    rounds = last_round

    # Initiate an empty dataframe to store our data
    all_championship_standings = pd.DataFrame()

    # We also want to store which driver drives for which team, which will help us later
    driver_team_mapping = {}
    driver_point_mapping = {}

    try:
        for i in range(1, rounds + 1):

            # Make request to driverStandings endpoint for the current round
            race = client.ergast_retrieve(f'{yr}/{i}/driverStandings')
            session = fastf1.get_session(yr, 1, 'R')

            # Get the standings from the result
            standings = race['StandingsTable']['StandingsLists'][0]['DriverStandings']

            # Initiate a dictionary to store the current rounds' standings in
            current_round = {'round': i}

            # Loop through all the drivers to collect their information
            for i in range(len(standings)):
                try:
                    driver = standings[i]['Driver']['code']
                except:
                    driver = " ".join(word[0].upper()+word[1:] for word in (
                        standings[i]['Driver']['driverId'].replace("_", " ")).split(" "))
                try:

                    position = standings[i]['position']
                except:
                    break
                points = standings[i]['points']

                # Store the drivers' position
                current_round[driver] = int(position)

                # Create mapping for driver-team to be used for the coloring of the lines
                driver_team_mapping[driver] = standings[i]['Constructors'][0]['name']

                driver_point_mapping[driver] = points
            all_championship_standings = pd.concat([all_championship_standings, pd.DataFrame(
                current_round, index=[0])], ignore_index=True)
    except IndexError:
        for i in range(1, rounds):

            # Make request to driverStandings endpoint for the current round
            race = client.ergast_retrieve(f'{yr}/{i}/driverStandings')
            session = fastf1.get_session(yr, 1, 'R')

            # Get the standings from the result
            standings = race['StandingsTable']['StandingsLists'][0]['DriverStandings']

            # Initiate a dictionary to store the current rounds' standings in
            current_round = {'round': i}

            # Loop through all the drivers to collect their information
            for i in range(len(standings)):
                try:
                    driver = standings[i]['Driver']['code']
                except:
                    driver = " ".join(word[0].upper()+word[1:] for word in (
                        standings[i]['Driver']['driverId'].replace("_", " ")).split(" "))
                try:

                    position = standings[i]['position']
                except:
                    break
                points = standings[i]['points']

                # Store the drivers' position
                current_round[driver] = int(position)

                # Create mapping for driver-team to be used for the coloring of the lines
                driver_team_mapping[driver] = standings[i]['Constructors'][0]['name']

                driver_point_mapping[driver] = points

            all_championship_standings = pd.concat([all_championship_standings, pd.DataFrame(
                current_round, index=[0])], ignore_index=True)

    rounds = i

    all_championship_standings = all_championship_standings.set_index('round')

    # Melt data so it can be used as input for plot
    all_championship_standings_melted = pd.melt(
        all_championship_standings.reset_index(), ['round'])

    # Initiate the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Set the title of the plot
    ax.set_title(str(yr) + " Championship Standing", color='white')

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    # Draw a line for every driver in the data by looping through all the standings
    # The reason we do it this way is so that we can specify the team color per driver
    for driver in pd.unique(all_championship_standings_melted['variable']):

        try:
            color = fastf1.plotting.get_team_color(
                driver_team_mapping[driver], session)
        except:
            color = colors[color_counter]
        sns.lineplot(
            x='round',
            y='value',
            data=all_championship_standings_melted.loc[all_championship_standings_melted['variable'] == driver],
            color=color
        )
        try:
            color = fastf1.plotting.get_team_color(
                driver_team_mapping[driver], session)
        except:
            color_counter += 1
            if color_counter >= len(colors):
                color_counter = 0

    # Invert Y-axis to have championship leader (#1) on top
    ax.invert_yaxis()

    # Set the values that appear on the x- and y-axes
    ax.set_xticks(range(1, max(schedule.RoundNumber)+1))
    if yr > 1995:
        ax.set_yticks(range(1, len(driver_team_mapping)+1))
    else:
        ax.set_yticks(range(1, 31))

    # set colorbar tick color
    ax.yaxis.set_tick_params(color='white')
    ax.xaxis.set_tick_params(color='white')

    # set colorbar ticklabels
    plt.setp(plt.getp(ax.axes, 'yticklabels'), color='white')
    plt.setp(plt.getp(ax.axes, 'xticklabels'), color='white')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    for t in ax.xaxis.get_ticklines():
        t.set_color('black')
    for t in ax.yaxis.get_ticklines():
        t.set_color('black')

    # Set the labels of the axes
    ax.set_xlabel("Round", color='white')
    ax.set_ylabel("Championship position", color='white')

    ax.grid(True, alpha=0.1)
    ax.minorticks_on()

    # Add the driver name to the lines
    for line, name, points in zip(ax.lines, all_championship_standings.columns.tolist(), driver_point_mapping.values()):
        y = line.get_ydata()[-1]
        x = line.get_xdata()[-1]

        text = ax.annotate(
            name + ": " + str(points),
            xy=(x + 0.1, y),
            xytext=(0, 0),
            color=line.get_color(),
            xycoords=(
                ax.get_xaxis_transform(),
                ax.get_yaxis_transform()
            ),
            textcoords="offset points"
        )

    file = utils.plot_to_file(fig, "plot")
    return file


def const_func(yr):
    import datetime
    schedule = fastf1.get_event_schedule(yr, include_testing=False)
    if yr == int(datetime.datetime.now().year):

        last_index = None
        for index, row in schedule.iterrows():

            if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                number = row['RoundNumber']
                last_index = number

        last_round = last_index
    else:
        last_round = max(schedule.RoundNumber)

    colors = ["#333333", "#444444", "#555555", "#666666", "#777777",
              "#888888", "#999999", "#AAAAAA", "#BBBBBB", "#CCCCCC"]
    color_counter = 0

    # Specify the number of rounds we want in our plot (in other words, specify the current round)
    rounds = last_round

    # Initiate an empty dataframe to store our data
    all_championship_standings = pd.DataFrame()

    # We also want to store which driver drives for which team, which will help us later
    constructor_team_mapping = {}
    constructor_point_mapping = {}

    try:
        for i in range(1, rounds + 1):
            try:
                # Make request to driverStandings endpoint for the current round
                race = client.ergast_retrieve(f'{yr}/{i}/constructorStandings')

                # Get the standings from the result
                standings = race['StandingsTable']['StandingsLists'][0]['ConstructorStandings']

                # Initiate a dictionary to store the current rounds' standings in
                current_round = {'round': i}

                # Loop through all the drivers to collect their information
                for i in range(len(standings)):
                    constructor = standings[i]['Constructor']['name']

                    position = standings[i]['position']

                    points = standings[i]['points']

                    # Store the drivers' position
                    current_round[constructor] = int(position)

                    # Create mapping for driver-team to be used for the coloring of the lines
                    constructor_team_mapping[constructor] = standings[i]['Constructor']['name']

                    constructor_point_mapping[constructor] = points

                # Append the current round to our fial dataframe
                all_championship_standings = pd.concat([all_championship_standings, pd.DataFrame(
                    current_round, index=[0])], ignore_index=True)
            except:
                break
    except IndexError:
        for i in range(1, rounds):
            try:
                # Make request to driverStandings endpoint for the current round
                race = client.ergast_retrieve(f'{yr}/{i}/constructorStandings')

                # Get the standings from the result
                standings = race['StandingsTable']['StandingsLists'][0]['ConstructorStandings']

                # Initiate a dictionary to store the current rounds' standings in
                current_round = {'round': i}

                # Loop through all the drivers to collect their information
                for i in range(len(standings)):
                    constructor = standings[i]['Constructor']['name']

                    position = standings[i]['position']

                    points = standings[i]['points']

                    # Store the drivers' position
                    current_round[constructor] = int(position)

                    # Create mapping for driver-team to be used for the coloring of the lines
                    constructor_team_mapping[constructor] = standings[i]['Constructor']['name']

                    constructor_point_mapping[constructor] = points

                # Append the current round to our fial dataframe
                all_championship_standings = pd.concat([all_championship_standings, pd.DataFrame(
                    current_round, index=[0])], ignore_index=True)
            except:
                break

    all_championship_standings = all_championship_standings.set_index('round')

    # Melt data so it can be used as input for plot
    all_championship_standings_melted = pd.melt(
        all_championship_standings.reset_index(), ['round'])

    rounds = i

    # Set the round as the index of the dataframe

    # Increase the size of the plot

    session = fastf1.get_session(yr, 1, "Race")
    session.load(laps=False, weather=False, telemetry=False, messages=False)
    # Initiate the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Set the title of the plot
    ax.set_title(str(yr) + " Championship Standing", color='white')

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    # Draw a line for every driver in the data by looping through all the standings
    # The reason we do it this way is so that we can specify the team color per driver
    for constructor in pd.unique(all_championship_standings_melted['variable']):
        try:
            color = fastf1.plotting.get_team_color(
                constructor_team_mapping[constructor], session)
        except:
            color = colors[color_counter]
        sns.lineplot(
            x='round',
            y='value',
            data=all_championship_standings_melted.loc[
                all_championship_standings_melted['variable'] == constructor],
            color=color
        )
        try:
            color = fastf1.plotting.get_team_color(
                constructor_team_mapping[constructor], session)
        except:
            color_counter += 1
            if color_counter >= len(colors):
                color_counter = 0

    # Invert Y-axis to have championship leader (#1) on top
    ax.invert_yaxis()

    # Set the values that appear on the x- and y-axes
    ax.set_xticks(range(1, max(schedule.RoundNumber)+1))
    ax.set_yticks(range(1, len(constructor_team_mapping)+1))

    # set colorbar tick color
    ax.yaxis.set_tick_params(color='white')
    ax.xaxis.set_tick_params(color='white')

    # set colorbar ticklabels
    plt.setp(plt.getp(ax.axes, 'yticklabels'), color='white')
    plt.setp(plt.getp(ax.axes, 'xticklabels'), color='white')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    for t in ax.xaxis.get_ticklines():
        t.set_color('black')
    for t in ax.yaxis.get_ticklines():
        t.set_color('black')

    # Set the labels of the axes
    ax.set_xlabel("Round", color='white')
    ax.set_ylabel("Championship position", color='white')

    ax.grid(True, alpha=0.1)
    ax.minorticks_on()

    # Add the driver name to the lines
    for line, name, points in zip(ax.lines, all_championship_standings.columns.tolist(), constructor_point_mapping.values()):
        y = line.get_ydata()[-1]
        x = line.get_xdata()[-1]

        text = ax.annotate(
            name.replace("amp;", "") + ": " + str(points),
            xy=(x + 0.1, y),
            xytext=(0, 0),
            color=line.get_color(),
            xycoords=(
                ax.get_xaxis_transform(),
                ax.get_yaxis_transform()
            ),
            textcoords="offset points"
        )

    # Save the plot
    # plt.show()
    file = utils.plot_to_file(fig, "plot")
    return file


async def h2h(year, session_type, ctx, include_dnfs):
    import datetime
    team_list = {}
    color_list = {}
    check_list = {}
    team_fullName = {}
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if session_type == "Sprint" and year < 2023:
        schedule = schedule[schedule['EventFormat'] == 'sprint']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint'
    elif session_type == "Sprint" and year == 2023:
        schedule = schedule[schedule['EventFormat'] == 'sprint_shootout']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint_shootout'
    elif session_type == "Sprint" and year >= 2024:
        schedule = schedule[schedule['EventFormat'] == 'sprint_qualifying']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint_qualifying'
    elif session_type == "Sprint Shootout":
        schedule = schedule[schedule['EventFormat'] == 'sprint_shootout']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint_shootout'
    elif session_type == "Sprint Qualifying":
        schedule = schedule[schedule['EventFormat'] == 'sprint_qualifying']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint_qualifying'
    else:
        schedule = schedule
        if year == int(datetime.datetime.now().year):
            max_index = roundnumber(None, year)[0]
        else:
            max_index = max(schedule.RoundNumber)
        result_setting = 'conventional'
        scheduleiteration = range(min(schedule.RoundNumber), max_index)
    if year < 2018:
        string = "LastName"
    else:
        string = "Abbreviation"
    tasks = []
    for c in scheduleiteration:

        event = await to_event(year, c)
        if result_setting in ['sprint_qualifying', 'sprint_shootout']:

            task = load_session(event, session_type, laps=True, telemetry=False,
                                weather=False, messages=True)
        else:
            task = load_session(event, session_type, laps=False, telemetry=False,
                                weather=False, messages=False)
        tasks.append(task)
    result = await asyncio.gather(*tasks)
    for i in result:
        try:
            results = i.results
            if results["DriverNumber"].size < len(i.drivers):
                raise Exception
        except:
            break
        for i in check_list.keys():
            check_list.update({i: False})

        for i in results['TeamName']:
            team_results = results.loc[lambda df: df['TeamName'] == i]

            if len(team_results.index) < 2:
                break
            if (team_list.get(i) is None):
                team_fullName.update(
                    {i: team_results.loc[min(team_results.index), 'TeamName']})
                team_list.update({i: {}})
                color_list.update(
                    {i: team_results.loc[min(team_results.index), 'TeamColor']})

            drivers = []
            for j in team_results.index:
                drivers.append(team_results.loc[j, string])
            try:
                drivers = sorted(drivers)
            except:
                raise Exception
            pairing = ''.join(drivers)

            if (team_list.get(i).get(pairing) is None):
                team_list.get(i).update({pairing: {}})

            for abbreviation in team_results[string]:
                if team_list.get(i).get(pairing).get(abbreviation) is None:
                    team_list.get(i).get(pairing).update({abbreviation: 0})

            curr_abbr = team_results.loc[team_results.index[0], string]

            both_drivers_finished = True
            if (session_type == 'Race' or session_type == 'Sprint'):

                dnf = ['D', 'E', 'W', 'F', 'N']
                if include_dnfs is False:
                    for driver in team_results.index:
                        if ((team_results.loc[driver, 'ClassifiedPosition']) in dnf) or (not ((team_results.loc[driver, 'Status'] == 'Finished') or ('+' in team_results.loc[driver, 'Status']))):
                            # for testing
                            # outstring += (f'{pairing}: Skipping {session}\nReason: {team_results.loc[driver,'Abbreviation']} did ({team_results.loc[driver,'ClassifiedPosition']},{team_results.loc[driver,'Status']})\n')
                            both_drivers_finished = False

            # if (team_list.get(i).get(pairing).get(curr_abbr) is None):
            #     team_list.get(i).get(pairing).update({curr_abbr:0})
            if (check_list.get(i) is None):
                check_list.update({i: False})
            if not check_list.get(i):
                curr_value = team_list.get(i).get(pairing).get(curr_abbr)
                # if include this race, then update the driver pairing's h2h "points"
                if (both_drivers_finished):
                    team_list.get(i).get(pairing).update(
                        {curr_abbr: curr_value+1})
                    check_list.update({i: True})
            else:
                curr_value = team_list.get(i).get(pairing).get(curr_abbr)
                team_list.get(i).get(pairing).update({curr_abbr: curr_value})
    data, colors, team_names = team_list, color_list, team_fullName
    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
    fig, ax = plt.subplots(1, figsize=(13, 9))

    fig.suptitle(f'{year} {session_type} Head to Head')
    offset = 0
    driver_names = []
    y_ticks = []
    for team in data.keys():
        for pairing in data.get(team).keys():
            y_ticks.append(team_names.get(team))
            drivers = list(data.get(team).get(pairing).keys())
            driver_wins = list(data.get(team).get(pairing).values())

            # flip second driver to draw back to back
            if len(driver_wins) >= 2:

                driver_wins[1] = -1 * driver_wins[1]

            else:
                driver_wins.append(0)

            # team color
            color = ''
            if not ((colors.get(team).lower() == 'nan') or (colors.get(team).lower() == '')):
                color = f'#{colors.get(team).lower()}'
            else:
                color = get_random_color()
            ax.barh(pairing, driver_wins, color=color,)  # edgecolor = 'black')

            # label the bars
            for i in range(len(drivers)):
                # Check if the driver participated
                if driver_wins[i] <= 0:
                    driver_name = drivers[i]
                    driver_names.append(driver_name)
                    wins_string = f'{-1*driver_wins[i]}'
                    ax.text(min(driver_wins[i] - 0.6, -1.2), offset - 0.2, wins_string,  fontsize=20,
                            horizontalalignment='right', path_effects=[matplotlib.patheffects.withStroke(linewidth=4, foreground="black")])
                else:
                    driver_name = drivers[i]
                    driver_names.append(driver_name)
                    wins_string = f'{driver_wins[i]}'
                    ax.text(driver_wins[i] + 0.6, offset - 0.2, wins_string,  fontsize=20,
                            horizontalalignment='left', path_effects=[matplotlib.patheffects.withStroke(linewidth=4, foreground="black")])
            offset += 1
    # plot formatting
    left = min(fig.subplotpars.left, 1 - fig.subplotpars.right)
    bottom = min(fig.subplotpars.bottom, 1 - fig.subplotpars.top)
    fig.subplots_adjust(left=left, right=1 - left,
                        bottom=bottom, top=1 - bottom)
    ax.get_xaxis().set_visible(False)
    ax.yaxis.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.set_yticklabels(y_ticks, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xabs_max = abs(max(ax.get_xlim(), key=abs))+7

    ax.text(0.1, 1.03, '', transform=ax.transAxes,
            fontsize=13, ha='center')

    ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)

    offset = 0
    # label drivers
    for i in range(len(driver_names)):
        if (i % 2) == 0:
            ax.text(xabs_max, offset-0.2, driver_names[i], fontsize=20, horizontalalignment='right', path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=4, foreground="black")])
        else:
            ax.text(-xabs_max, math.floor(offset)-0.2, driver_names[i], fontsize=20, horizontalalignment='left', path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=4, foreground="black")])
        offset += 0.5
    plt.rcParams['savefig.dpi'] = 300

    file = utils.plot_to_file(fig, "image")
    top_role_color = get_top_role_color(ctx.author)
    title = f"Teammate {session_type} Head to Head {year}"
    description = ""

    return customEmbed(title=title, description=description, image_url='attachment://image.png', colour=top_role_color), file


async def averageposition(session_type, year, category, ctx, include_dnfs):

    import datetime
    schedule = fastf1.get_event_schedule(year=year, include_testing=False)
    if session_type == "Sprint" and year < 2023:
        schedule = schedule[schedule['EventFormat'] == 'sprint']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint'
    elif session_type == "Sprint" and year == 2023:
        schedule = schedule[schedule['EventFormat'] == 'sprint_shootout']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint_shootout'
    elif session_type == "Sprint" and year >= 2024:
        schedule = schedule[schedule['EventFormat'] == 'sprint_qualifying']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint_qualifying'
    elif session_type == "Sprint Shootout":
        schedule = schedule[schedule['EventFormat'] == 'sprint_shootout']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint_shootout'
    elif session_type == "Sprint Qualifying":
        schedule = schedule[schedule['EventFormat'] == 'sprint_qualifying']
        scheduleiteration = schedule['RoundNumber'].tolist()
        result_setting = 'sprint_qualifying'
    else:
        schedule = schedule
        if year == int(datetime.datetime.now().year):
            max_index = roundnumber(None, year)[0]
        else:
            max_index = max(schedule.RoundNumber)
        result_setting = 'conventional'
        scheduleiteration = range(min(schedule.RoundNumber), max_index)
    driver_positions = {}
    driver_average = {}
    driver_colors = {}
    driver_racesParticipated = {}
    tasks = []

    for i in scheduleiteration:
        event = await to_event(year, i)
        if result_setting in ['sprint_qualifying', 'sprint_shootout']:

            task = load_session(event, session_type, laps=True, telemetry=False,
                                weather=False, messages=True)
        else:
            task = load_session(event, session_type, laps=False, telemetry=False,
                                weather=False, messages=False)
        tasks.append(task)
    result = await asyncio.gather(*tasks)
    for i in result:
        results = i.results
        results.dropna(subset=['Position'], inplace=True)

        for driver in i.drivers:
            if (category == 'Drivers'):
                currDriver_abbreviation = results.loc[driver, 'LastName']
            else:
                currDriver_abbreviation = results.loc[driver, 'TeamName']
            try:
                if driver_positions.get(currDriver_abbreviation) is None:
                    driver_positions.update({currDriver_abbreviation: 0})
            except:
                break

            if driver_racesParticipated.get(currDriver_abbreviation) is None:
                driver_racesParticipated.update(
                    {currDriver_abbreviation: 0})

            if (session_type == 'Race' or session_type == 'Sprint') and include_dnfs is True:
                currDriver_position = results.loc[driver, 'Position']
            elif (session_type == 'Race' or session_type == 'Sprint') and include_dnfs is False:
                currDriver_position = results.loc[driver, 'ClassifiedPosition']
            else:
                currDriver_position = results.loc[driver, 'Position']

            currDriver_total = driver_positions.get(
                currDriver_abbreviation)

            if (type(currDriver_position) is str):
                if (currDriver_position.isnumeric()):
                    driver_racesParticipated.update(
                        {currDriver_abbreviation: driver_racesParticipated.get(currDriver_abbreviation)+1})
                    driver_positions.update(
                        {currDriver_abbreviation: currDriver_total+int(currDriver_position)})
            else:
                driver_racesParticipated.update(
                    {currDriver_abbreviation: driver_racesParticipated.get(currDriver_abbreviation)+1})
                driver_positions.update(
                    {currDriver_abbreviation: currDriver_total+(currDriver_position)})

            driver_colors.update(
                {currDriver_abbreviation: results.loc[driver, 'TeamColor']})
    for key in driver_positions.keys():
        try:
            driver_average.update({key: driver_positions.get(
                key)/driver_racesParticipated.get(key)})
        except:
            print('div by 0')
    driver_positions, driver_colors = driver_average, driver_colors
    driver_positions = dict(
        sorted(driver_positions.items(), key=lambda x: x[1]))

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
    fig, ax = plt.subplots(figsize=(16.8, 10.5))

    # setting x-axis label, title
    ax.set_xlabel("Position", fontsize=20, labelpad=20)
    ax.set_title(
        f"Average {category} {session_type} Finish Position {year}", fontsize=20, pad=20)

    # space between limits for the y-axis and x-axis
    ax.set_ylim(-0.8, len(driver_positions.keys())-0.25)
    ax.set_xlim(0, 20.1)
    ax.invert_yaxis()  # invert y axis, top to bottom
    # amount x-axis increments by 1
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # remove ticks, keep labels
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticklabels(
        range(-1, int(max(driver_positions.values()))+6),  fontsize=20)
    if (category == 'Drivers'):
        fontsize = 20
    else:
        fontsize = 10
    ax.set_yticklabels(driver_positions.keys(), fontsize=fontsize)
    ax.set_yticks(range(len(driver_positions.keys())))
    ax.tick_params(axis='both', length=0, pad=8, )

    # remove all lines, bar the x-axis grid lines
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color='#191919')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(which="minor", alpha=0.1)
    ax.minorticks_on()

    annotated = False
    for driver in driver_positions.keys():
        curr_color = driver_colors.get(driver)
        if ((curr_color != 'nan') and (curr_color != '')):
            plt.barh(driver, driver_positions.get(
                driver), color=f'#{curr_color}')
        else:

            if not annotated:
                annotated = True
            plt.barh(driver, driver_positions.get(
                driver), color=get_random_color())

    for i, position in enumerate(driver_positions.values()):
        ax.text(position + 0.1, i,
                f"   {str(round(position,2))}", va='center',  fontsize=20)
    file = utils.plot_to_file(fig, "image")
    description = ""
    title = f"Average {category} {session_type} Finish Position {year}"
    top_role_color = get_top_role_color(ctx.author)

    return customEmbed(title=title, description=description, image_url='attachment://image.png', colour=top_role_color), file


def roundnumber(round=None, year=None):
    import datetime
    if round == None and year == None or round == None and year == int(datetime.datetime.now().year):

        schedule = fastf1.get_event_schedule(
            int(datetime.datetime.now().year), include_testing=False)

        for index, row in schedule.iterrows():

            if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
                number = row['RoundNumber']
                round = number
    if year == None:
        year = int(datetime.datetime.now().year)
    if year < int(datetime.datetime.now().year) and round == None:
        round = max(fastf1.get_event_schedule(
            int(year), include_testing=False)['RoundNumber'])
    return [round, year]


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
        if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
            number = row['RoundNumber']
            next_event = number+1
    out_string = "Season has finished or hasn't begun yet!"
    try:
        if (len(schedule) < next_event):
            raise IndexError

        race_name = schedule.loc[next_event, "EventName"]

        message_embed.title = "Race Schedule for " + race_name

        if (schedule.loc[next_event, "EventFormat"] == 'conventional'):
            converted_session_times = {
                f":one: {schedule.loc[next_event, 'Session1']}": schedule.loc[next_event, "Session1Date"],
                f":two: {schedule.loc[next_event, 'Session2']}": schedule.loc[next_event, "Session2Date"],
                f":three: {schedule.loc[next_event, 'Session3']}": schedule.loc[next_event, "Session3Date"],
                f":stopwatch: {schedule.loc[next_event, 'Session4']}": schedule.loc[next_event, "Session4Date"],
                f":checkered_flag: {schedule.loc[next_event, 'Session5']}": schedule.loc[next_event, "Session5Date"]
            }

        else:
            converted_session_times = {
                f":one: {schedule.loc[next_event, 'Session1']}": schedule.loc[next_event, "Session1Date"],
                f":stopwatch: {schedule.loc[next_event, 'Session2']}": schedule.loc[next_event, "Session2Date"],
                f":stopwatch: {schedule.loc[next_event, 'Session3']}": schedule.loc[next_event, "Session3Date"],
                f":race_car: {schedule.loc[next_event, 'Session4']}": schedule.loc[next_event, "Session4Date"],
                f":checkered_flag: {schedule.loc[next_event, 'Session5']}": schedule.loc[next_event, "Session5Date"]
            }

        location = schedule.loc[next_event, "Location"]
        sessions_string = ''
        times_string = ''

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

        return message_embed
    except IndexError:
        return out_string


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


def get_fia_doc(year, eventname, doc=None, doc_name=None, get_all_docs=False):

    message_embed = discord.Embed()
    message_embed.title = f"FIA Document {doc}"

    if doc is None:
        doc = 0
        message_embed.title = "Latest FIA Document"
    base_url = "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-"
    url = base_url + year_mapping.get(year)
    if eventname == '70th Anniversary Grand Prix':
        eventname = 'Formula 1 70th Anniversary Grand Prix'
    elif eventname == 'Mexico City Grand Prix' and year < 2023:
        eventname = 'Mexican Grand Prix'
    elif eventname == 'São Paulo Grand Prix' and year < 2023:
        eventname = 'Brazilian Grand Prix'
    elif eventname == 'Saudi Arabian Grand Prix' and year < 2023:
        eventname = 'Saudi Arabia Grand Prix'

    if eventname is None:
        pass
    else:
        url = url + f"/event/{eventname.replace(' ', '%20')}"
    html = requests.get(url=url)
    s = BeautifulSoup(html.content, 'html.parser')

    results = s.find_all(class_='document-row')
    documents = [result.find('a')['href']
                 for result in results if result.find('a')]
    if get_all_docs:
        return documents
    else:
        if doc_name:
            doc_url = 'https://www.fia.com/' + doc_name

            fileName = doc_name.split(
                '/sites/default/files/decision-document/')[-1]
        else:
            doc_url = 'https://www.fia.com/' + documents[doc]
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
                img = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples)
                img.save(img_stream, format="PNG")
                img_stream.seek(0)  # Go to the beginning of the BytesIO stream
                images.append(discord.File(
                    img_stream, filename=f"{page_num}.png"))
                page_num += 1
        except ValueError:
            pass
        doc.close()

        return images


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
            return message_embed
    except:
        print('Error occured!')


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

    SEASON = int(datetime.now().year)
    ROUND = last_index
    ergast = Ergast()
    standings = ergast.get_driver_standings(season=SEASON, round=ROUND)
    if standings.content == []:  # check for late updates of ergast data
        standings = ergast.get_driver_standings(season=SEASON, round=ROUND-1)

    return standings.content[0]


def calculate_max_points_for_remaining_season():

    schedule = fastf1.get_event_schedule(
        int(datetime.now().year), include_testing=False)
    last_index = None
    for index, row in schedule.iterrows():
        if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
            last_index = index

    SEASON = int(datetime.now().year)
    ROUND = last_index
    POINTS_FOR_SPRINT = 8 + 25
    POINTS_FOR_CONVENTIONAL = 25
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


def get_ephemeral_setting(ctx: ApplicationContext) -> bool:
    try:
        default_ephemeral = True
        if ctx.guild.name is not None:
            guild_id = ctx.guild_id
            conn = sqlite3.connect("bot_settings.db")
            cursor = conn.cursor()

            try:
                cursor.execute(
                    "SELECT ephemeral_setting FROM settings WHERE guild_id = ?", (guild_id,))
                result = cursor.fetchone()
                if result is not None:
                    # Return the fetched ephemeral setting
                    return bool(result[0])
                else:
                    return default_ephemeral  # Guild not found, return default
            except Exception as e:
                return default_ephemeral  # On error, return default
            finally:
                conn.close()
        else:
            return True
    except:
        return True


def get_session_type(name: str):
    """Return one of `["R", "Q", "P"]` depending on session `name`.

    E.g. "Race/Sprint" is type "R".
    "Qualifying/Sprint Shootout" is type "Q".
    """
    if "Practice" in name:
        return "P"
    if name in ("Qualifying", "Sprint Shootout", "Sprint Qualifying"):
        return "Q"
    return "R"


def plot_table(df: pd.DataFrame, col_defs: list[ColDef], idx: str, figsize: tuple[float]):
    """Returns plottable table from data."""

    fig = Figure(figsize=figsize, dpi=200, layout="constrained")
    ax = fig.add_subplot()

    table = Table(
        df=df,
        ax=ax,
        index_col=idx,
        textprops={"fontsize": 10, "ha": "center"},
        column_border_kw={"color": fig.get_facecolor(), "lw": 2},
        col_label_cell_kw={"facecolor": (0, 0, 0, 0.35)},
        col_label_divider_kw={"color": fig.get_facecolor(), "lw": 4},
        row_divider_kw={"color": fig.get_facecolor(), "lw": 1.2},
        column_definitions=col_defs,
    )
    table.col_label_row.set_fontsize(11)
    del df

    return table


def plot_race_control_table(df: pd.DataFrame, col_defs: list[ColDef], idx: str, figsize: tuple[float]):
    """Returns plottable table from data."""

    fig = Figure(figsize=figsize, dpi=200, layout="tight")
    ax = fig.add_subplot()

    table = Table(
        df=df,
        ax=ax,
        index_col=idx,
        textprops={"fontsize": 10, "ha": "center"},
        column_border_kw={"color": fig.get_facecolor(), "lw": 2},
        col_label_cell_kw={"facecolor": (0, 0, 0, 0.35)},
        col_label_divider_kw={"color": fig.get_facecolor(), "lw": 4},
        row_divider_kw={"color": fig.get_facecolor(), "lw": 1.2},
        column_definitions=col_defs,
    )
    table.col_label_row.set_fontsize(11)
    del df

    return table


async def to_event(year: str, rnd: str) -> Event:
    """Get a `fastf1.events.Event` for a race weekend corresponding to `year` and `round`.

    Handles conversion of "last" round and "current" season from Ergast API.

    The `round` can also be a GP name or circuit.
    """
    # Get the actual round id from the last race endpoint
    if rnd == "last":
        data = await ergast.race_info(year, "last")
        rnd = data["round"]

    if str(rnd).isdigit():
        rnd = int(rnd)

    try:
        event = await asyncio.to_thread(fastf1.get_event, year=utils.convert_season(year), gp=rnd)
    except Exception:
        raise MissingDataError()

    return event


async def load_session(event: Event, name: str, **kwargs) -> Session:
    """Searches for a matching `Session` using `name` (session name, abbreviation or number).

    Loads and returns the `Session`.
    """
    try:
        # Run FF1 blocking I/O in async thread so the bot can await
        session = await asyncio.to_thread(event.get_session, identifier=name)
        await asyncio.to_thread(session.load,
                                laps=kwargs.get("laps", False),
                                telemetry=kwargs.get('telemetry', False),
                                weather=kwargs.get("weather", False),
                                messages=kwargs.get("messages", False),
                                livedata=kwargs.get("livedata", None))
    except Exception:
        raise MissingDataError(
            "Unable to get session data, check the round and year is correct.")

    finally:
        gc.collect()

    return session


async def format_results(session: Session, name: str, year):
    """Format the data from `Session` results with data pertaining to the relevant session `name`.

    The session should be already loaded.

    Returns
    ------
    `DataFrame` with columns:

    Qualifying / Sprint Shootout - `[Pos, Code, Driver, Team, Q1, Q2, Q3]` \n
    Race / Sprint - `[Pos, Code, Driver, Team, Grid, Finish, Points, Status]` \n
    Practice - `[Code, Driver, Team, Fastest, Laps]`
    """

    _session_type = get_session_type(name)

    # Handle missing results data
    try:
        _sr: SessionResults = session.results
        if _sr["DriverNumber"].size < len(session.drivers):
            raise Exception
        if not _session_type == "P" and np.all(_sr["Position"].isna().values):
            raise Exception
        if _session_type == "R":
            _sr.dropna(subset=['Position'], inplace=True)
            _sr.reset_index(drop=True)

    except Exception:
        raise MissingDataError(
            "Session data unavailable. If the session finished recently, check again later."
        )

    if year < 2018:
        res_df: SessionResults = _sr.rename(columns={
            "Position": "Pos",
            "DriverNumber": "No",
            "Abbreviation": "Code",
            "FullName": "Driver",
            "GridPosition": "Grid",
            "TeamName": "Team"
        })
        del _sr
    else:
        res_df: SessionResults = _sr.rename(columns={
            "Position": "Pos",
            "DriverNumber": "No",
            "Abbreviation": "Code",
            "BroadcastName": "Driver",
            "GridPosition": "Grid",
            "TeamName": "Team"
        })
        del _sr

    res_df['Driver'] = res_df['Driver'].str.replace("_", " ")
    res_df['Driver'] = res_df['Driver'].str.title()

    if _session_type == "P":
        # Reload the session to fetch missing lap info
        await asyncio.to_thread(session.load, laps=True, telemetry=False,
                                weather=False, messages=False, livedata=None)

        # Get each driver's fastest lap in the session
        fastest_laps = session.laps.groupby("DriverNumber")["LapTime"] \
            .min().reset_index().set_index("DriverNumber")

        # Combine the fastest lap data with the results data
        fp = pd.merge(
            res_df[["Code", "Driver", "Team"]],
            fastest_laps["LapTime"],
            left_index=True, right_index=True)

        del fastest_laps, res_df

        # Get a count of lap entries for each driver
        lap_totals = session.laps.groupby("DriverNumber").count()
        fp["Laps"] = lap_totals["LapNumber"]

        # Format the lap timedeltas to strings
        fp["LapTime"] = fp["LapTime"].apply(
            lambda x: utils.format_timedelta(x))
        fp = fp.rename(columns={"LapTime": "Fastest"}
                       ).sort_values(by="Fastest")

        return fp

    if _session_type == "Q":
        res_df["Pos"] = res_df["Pos"].astype(int)
        qs_res = res_df.loc[:, ["Pos", "Code",
                                "Driver", "Team", "Q1", "Q2", "Q3"]]

        # Format the timedeltas to readable strings, replacing NaT with blank
        qs_res.loc[:, ["Q1", "Q2", "Q3"]] = res_df.loc[:, [
            "Q1", "Q2", "Q3"]].applymap(lambda x: utils.format_timedelta(x))

        del res_df
        return qs_res

    # Get leader finish time
    leader_time = res_df["Time"].iloc[0]

    # Format the Time column:
    # Leader finish time; followed by gap in seconds to leader
    # Drivers who were a lap behind or retired show the finish status instead, e.g. '+1 Lap' or 'Collision'
    res_df["Finish"] = res_df.apply(lambda r: f"+{r['Time'].total_seconds():.3f}"
                                    if r['Status'] == 'Finished' else r['Status'], axis=1)

    # Format the timestamp of the leader lap
    res_df.loc[res_df.first_valid_index(), "Finish"] = utils.format_timedelta(
        leader_time, hours=True)

    res_df["Pos"] = res_df["Pos"].astype(int)
    res_df["Pts"] = res_df["Points"].astype(int)
    res_df["Grid"] = res_df["Grid"].astype(int)

    return res_df.loc[:, ["Pos", "Driver", "Team", "Grid", "Finish", "Pts", "Status"]]


async def filter_pitstops(year, round, s=None, filter: str = None, driver: str = None) -> pd.DataFrame:
    drv_lst = await ergast.get_all_drivers(year, round)
    drv_info = {d["driverId"]: d for d in drv_lst}
    drv_codes = {d['code']: d for d in drv_lst}
    drv_id_mapping = {d['driverId']: d['code'] for d in drv_lst}
    drv_laps = []
    if driver is not None:
        driver = await asyncio.to_thread(lambda: utils.find_driver(driver, drv_lst)["driverId"])
        if s:
            box_laps = await asyncio.to_thread(lambda:  s.laps.pick_drivers(utils.find_driver(
                driver, drv_lst)['code']).pick_box_laps())
            for i in range(0, len(box_laps['PitInTime'].dropna(inplace=False))):
                try:
                    box_laps2 = box_laps.dropna(subset=['PitInTime'])
                    pit_in_time = box_laps['PitInTime'].dropna(
                        inplace=False).iloc[i]
                    lap_number = box_laps2['LapNumber'].iloc[i]
                    pit_out_time = box_laps['PitOutTime'].dropna(
                        inplace=False).iloc[i]
                    accurate_lap = await asyncio.to_thread(lambda: box_laps.get_car_data().fill_missing())
                    stationary_period = accurate_lap.slice_by_time(
                        pit_in_time, pit_out_time)
                    stationary_period = stationary_period[stationary_period['Speed'] == 0]
                    total_stationary_time = (max(stationary_period['SessionTime']) -
                                             min(stationary_period['SessionTime'])).total_seconds()+0.239  # adding time for better accuracy

                    if total_stationary_time > 1.8:
                        drv_laps.append([lap_number, total_stationary_time])
                except:
                    continue

    def get_stationary_time(row):
        import builtins
        lap = row['lap']
        for lap_data in drv_laps:
            if str(int(lap_data[0])) == str(lap):
                return builtins.round(lap_data[1], 3)
            else:
                if str(int(lap_data[0])) == str(int(lap)+1):
                    return builtins.round(lap_data[1], 3)

    def get_stationary_time2(row):
        import builtins
        driver = drv_id_mapping.get(row['driverId'])
        lap = row['lap']
        if driver in drv_lap:
            for lap_data in drv_lap[driver]:
                if str(int(lap_data[0])) == str(lap):
                    return builtins.round(lap_data[1], 3)
                else:
                    if str(int(lap_data[0])) == str(int(lap)+1):
                        str(int(lap_data[0])) == str(int(lap)+1)
                        return builtins.round(lap_data[1], 3)

    res = await asyncio.to_thread(
        ff1_erg.get_pit_stops,
        season=year, round=round,
        driver=driver, limit=1000)

    data = res.content[0]
    if driver is not None and s is not None:
        row_mask = data["driverId"] == driver
        data['Stationary Time'] = data.apply(get_stationary_time, axis=1)
        data.dropna(inplace=True)
    elif driver is not None and s is None:
        row_mask = data["driverId"] == driver
    elif driver is None and s is not None:
        drv_lap = {}

        for j in drv_codes.keys():
            box_laps = await asyncio.to_thread(lambda: s.laps.pick_drivers(j).pick_box_laps())
            for i in range(0, len(box_laps['PitInTime'].dropna(inplace=False))):
                try:
                    drv = box_laps['Driver'].iloc[i]
                    if drv not in drv_lap:
                        drv_lap[drv] = []
                    box_laps2 = box_laps.dropna(subset=['PitInTime'])
                    pit_in_time = box_laps['PitInTime'].dropna(
                        inplace=False).iloc[i]
                    lap_number = box_laps2['LapNumber'].iloc[i]
                    pit_out_time = box_laps['PitOutTime'].dropna(
                        inplace=False).iloc[i]
                    stationary_period = await asyncio.to_thread(lambda: box_laps.get_car_data().slice_by_time(pit_in_time, pit_out_time))
                    stationary_period = stationary_period[stationary_period['Speed'] == 0]
                    total_stationary_time = (max(stationary_period['SessionTime']) -
                                             min(stationary_period['SessionTime'])).total_seconds()+0.239
                    if total_stationary_time > 1.8:
                        drv_lap[j].append([lap_number, total_stationary_time])

                except:
                    continue

        row_mask = data.groupby("driverId")["duration"].idxmin()
        data['Stationary Time'] = data.apply(get_stationary_time2, axis=1)

    else:
        row_mask = data.groupby("driverId")["duration"].idxmin()
    if year > 2017:
        df = data
        df = df.sort_values(
            by="Stationary Time").reset_index(drop=True)
        df.dropna(inplace=True)
    else:
        df = data.loc[row_mask].sort_values(
            by="duration").reset_index(drop=True)

    del data
    df["duration"] = df["duration"].transform(
        lambda x: f"{x.total_seconds():.3f}")

    df["Code"] = df.apply(lambda x: pd.Series({
        "Code": drv_info[x.driverId]["code"]
    }), axis=1)

    if filter.lower() == "best":
        if year > 2017:
            df = df.loc[[df["Stationary Time"].astype(float).idxmin()]]
        else:
            df = df.loc[[df["duration"].astype(float).idxmin()]]

    if filter.lower() == "worst":
        if year > 2017:
            df = df.loc[[df["Stationary Time"].astype(float).idxmax()]]
        else:
            df = df.loc[[df["duration"].astype(float).idxmax()]]
    df.columns = df.columns.str.title()
    if year > 2017:
        return df.loc[:, ["Code", "Stop", "Lap", "Duration", "Stationary Time"]]
    else:
        return df.loc[:, ["Code", "Stop", "Lap", "Duration"]]


async def tyre_stints(session: Session, driver: str = None):
    """Return a DataFrame showing each driver's stint on a tyre compound and
    the number of laps driven on the tyre.

    The `session` must be a loaded race session with laps data.

    Raises
    ------
        `MissingDataError`: if session does not support the API lap data
    """

    # Group laps data to individual sints per compound with total laps driven
    stints = session.laps.loc[:, ["Driver", "Stint",
                                  "Compound", 'FreshTyre', "TyreLife", 'LapNumber']]

    stints = stints.groupby(["Driver", "Stint", "Compound", 'FreshTyre']).agg(
        Lap=("LapNumber", "first"),
        TyreLife=("TyreLife", "last")
    ).reset_index()

    stints["Stint"] = stints["Stint"].astype(int)

    # Try to find the driver if given and filter results
    if driver is not None:
        year, rnd = session.event["EventDate"].year, session.event["RoundNumber"]
        drv_code = utils.find_driver(driver, await ergast.get_all_drivers(year, rnd))["code"]

        return stints.loc[stints["Driver"] == drv_code].set_index(["Driver", "Stint"], drop=True)

    return stints


def minisectors(laps: list[Lap]) -> pd.DataFrame:
    """Get driver telemetry and calculate the minisectors for each data row based on distacne.

    The `laps` should be loaded from the session.

    Returns
    ------
        `DataFrame`: [Driver, Time, Distance, Speed, mSector, X, Y]
    """

    tel_list = []

    # Load telemetry for the laps
    for lap in laps:
        t = lap.get_telemetry()
        t["Driver"] = lap["Driver"]
        tel_list.append(t)

    # Create single df with all telemetry
    telemetry = pd.concat(tel_list).reset_index(drop=True)
    del tel_list

    # Assign minisectors to each row based on distance
    max_dis = telemetry["Distance"].values.max()
    ms_len = max_dis / 24
    telemetry["mSector"] = telemetry["Distance"].apply(lambda x: (
        int((x // ms_len) + 1)
    ))

    return telemetry.loc[:, ["Driver", "Time", "Distance", "Speed", "X", "Y", "mSector"]]


def team_pace(session: Session):
    """Get the max sector speeds and min sector times from the lap data for each team in the session.

    The `session` must be loaded with laps data.

    Returns
    ------
        `DataFrame` containing max sector speeds and avg times indexed by team.

    Raises
    ------
        `MissingDataError`: if session doesn't support lap data.
    """

    # Get only the quicklaps in session to exclude pits and slow laps
    laps = session.laps.pick_quicklaps()
    times = laps.groupby(["Team"])[
        ["Sector1Time", "Sector2Time", "Sector3Time"]].mean()
    speeds = laps.groupby(["Team"])[
        ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].max()
    del laps

    df = pd.merge(times, speeds, how="left", left_index=True, right_index=True)

    return df


async def fastest_laps(session: Session, tyre: str = None):
    """Get fastest laptimes for all drivers in the session, optionally filtered by `tyre`.

    Returns
    ------
        `DataFrame` [Rank, Driver, LapTime, Delta, Lap, Tyre, ST]

    Raises
    ------
        `MissingDataError` if lap data unsupported or no lap data for the tyre.
    """

    laps = await asyncio.to_thread(lambda: session.laps.pick_wo_box())

    if tyre:
        laps = await asyncio.to_thread(lambda: laps.pick_compounds(tyre))

    if laps["Driver"].size == 0:
        raise MissingDataError("Not enough laps on this tyre.")

    fastest = Laps(
        [laps.pick_drivers(d).pick_fastest()
         for d in laps["Driver"].dropna().unique()]
    ).sort_values(by="LapTime").reset_index(drop=True).rename(
        columns={
            "LapNumber": "Lap",
            "Compound": "Tyre",
            "SpeedST": "ST"
        }
    )
    del laps
    fastest["Delta"] = fastest["LapTime"] - fastest["LapTime"].min()
    fastest["Rank"] = np.arange(1, fastest.index.size + 1)
    fastest["LapTime"] = fastest["LapTime"].apply(
        lambda x: utils.format_timedelta(x))
    fastest[["Lap", "ST"]] = fastest[["Lap", "ST"]].fillna(0.0).astype(int)
    fastest.dropna(subset=['Tyre'], inplace=True)

    return fastest.loc[:, ["Rank", "Driver", "LapTime", "Delta", "Lap", "Tyre", "ST"]]


async def sectors(s: Session, tyre: str = None):
    """Get a DataFrame showing the minimum sector times and max speed trap recorded for each driver.
    Based on quicklaps only. Optionally filter by tyre compound.

    Parameters
    ------
        `s`: a loaded race session

    Returns
    ------
        `DataFrame` [Driver, S1, S2, S3, ST]

    Raises
    ------
        `MissingDataError` if lap data not supported or not enough laps with tyre compound.
    """

    # Get quicklaps
    laps = await asyncio.to_thread(lambda: s.laps.pick_quicklaps())

    # Filter by tyre if chosen
    if tyre:
        laps = await asyncio.to_thread(lambda: laps.pick_compounds(tyre))

    if laps["Driver"].size == 0:
        raise MissingDataError("No quick laps available for this tyre.")

    # Get finish order for sorting
    finish_order = pd.DataFrame(
        {"Driver": s.results["Abbreviation"].values}
    ).set_index("Driver", drop=True)

    # Max speed for each driver
    speeds = laps.groupby("Driver")["SpeedST"].max(
    ).reset_index().set_index("Driver", drop=True)

    # Min sectors
    sectors = laps.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]] \
        .min().reset_index().set_index("Driver", drop=True)
    sectors["ST"] = speeds["SpeedST"].astype(int)

    # Merge with the finish order to get the data sorted
    df = pd.merge(finish_order, sectors, left_index=True,
                  right_index=True).reset_index()
    # Convert timestamps to seconds
    df[["S1", "S2", "S3"]] = df[
        ["Sector1Time", "Sector2Time", "Sector3Time"]
    ].applymap(lambda x: f"{x.total_seconds():.3f}")

    return df


def tyre_performance(session: Session):
    """Get a DataFrame showing the average lap times for each tyre compound based on the
    number of laps driven on the tyre.

    Data is grouped by Compound and TyreLife to get the average time for each lap driven.
    Lap time values are based on quicklaps using a threshold of 105%.

    Parameters
    ------
        `session` should already be loaded with lap data.

    Returns
    ------
        `DataFrame` [Compound, TyreLife, LapTime, Seconds]
    """

    # Filter and group quicklaps within 105% by Compound and TyreLife to get the mean times per driven lap
    laps = session.laps.pick_quicklaps(1.05).groupby(["Compound", "TyreLife"])[
        "LapTime"].mean().reset_index()
    laps["Seconds"] = laps["LapTime"].dt.total_seconds()

    return laps


def pos_change(session: Session):
    """Returns each driver start, finish position and the difference between them. Session must be race."""

    if session.name != "Race":
        raise MissingDataError("The session should be race.")

    diff = session.results.loc[:, ["Abbreviation", "GridPosition", "Position"]].rename(
        columns={
            "Abbreviation": "Driver",
            "GridPosition": "Start",
            "Position": "Finish"
        }
    ).reset_index(drop=True).sort_values(by="Finish")

    diff["Diff"] = diff["Start"] - diff["Finish"]
    diff[["Start", "Finish", "Diff"]] = diff[[
        "Start", "Finish", "Diff"]].astype(int)

    return diff


def get_dnf_results(session: Session):
    """Filter the results to only drivers who retired and include their final lap."""

    driver_nums = [
        d for d in session.drivers
        if (
            session.laps.pick_drivers(
                d)["LapNumber"].isna().all()  # No lap data
            # Retired before max lap
            or session.laps.pick_drivers(d)["LapNumber"].astype(int).max() < session.race_control_messages['Lap'].max()
        )
    ]

    dnfs = session.results.loc[session.results["DriverNumber"].isin(
        driver_nums)].reset_index(drop=True)

    dnfs["LapNumber"] = [session.laps.pick_drivers(
        d)["LapNumber"].astype(int).max() for d in driver_nums]
    dnfs = dnfs[~dnfs["Status"].isin(
        ["Finished", "+1 Lap"])].reset_index(drop=True)

    return dnfs


def results_table(results: pd.DataFrame, name: str) -> tuple[Figure, Axes]:
    """Return a formatted matplotlib table from a session results dataframe.

    `name` is the session name parameter.
    """
    base_defs = [
        ColDef(name="Driver", width=0.9, textprops={"ha": "left"}),
        ColDef(name="Team", width=0.8, textprops={"ha": "left"}),
    ]
    pos_def = ColDef("Pos", width=0.5, textprops={
                     "weight": "bold"}, border="right")

    if get_session_type(name) == "R":
        size = (8.5, 10)
        idx = "Pos"
        dnfs = results.loc[~results["Status"].isin(
            ["+1 Lap", "Finished"]), "Pos"].astype(int).values
        results = results.drop("Status", axis=1)
        results = results[~results.index.duplicated(keep='first')]
        col_defs = base_defs + [
            pos_def,
            ColDef(name="Code", width=0.4),
            ColDef("Grid", width=0.35),
            ColDef("Pts", width=0.35, border="l"),
            ColDef("Finish", width=0.66, textprops={
                   "ha": "right"}, border="l"),
        ]

    if get_session_type(name) == "Q":
        size = (10, 10)
        idx = "Pos"
        col_defs = base_defs + [pos_def, ColDef(name="Code", width=0.4)] + [
            ColDef(n, width=0.5) for n in ("Q1", "Q2", "Q3")
        ]

    if get_session_type(name) == "P":
        size = (8, 10)
        idx = "Code"
        col_defs = base_defs + [
            ColDef("Code", width=0.4, textprops={
                   "weight": "bold"}, border="right"),
            ColDef("Fastest", width=0.5, textprops={"ha": "right"}),
            ColDef("Laps", width=0.35, textprops={"ha": "right"}),
        ]

    table = plot_table(df=results, col_defs=col_defs, idx=idx, figsize=size)
    del results

    if get_session_type(name) == "R":
        for i in dnfs:
            table.rows[i - 1].set_facecolor((0, 0, 0, 0.38)
                                            ).set_hatch("//").set_fontcolor((1, 1, 1, 0.5))

    return table.figure, table.ax


def pitstops_table(results: pd.DataFrame, year) -> tuple[Figure, Axes]:
    """Returns matplotlib table from pitstops results DataFrame."""
    if year > 2017:
        col_defs = [
            ColDef("Code", width=0.4, textprops={
                   "weight": "bold"}, border="r"),
            ColDef("Stop", width=0.25),
            ColDef("Lap", width=0.25),
            ColDef("Duration", width=0.5),
            ColDef("Stationary Time", width=0.5,
                   textprops={"ha": "right"}, border="l")

        ]
    else:
        col_defs = [
            ColDef("Code", width=0.4, textprops={
                   "weight": "bold"}, border="r"),
            ColDef("Stop", width=0.25),
            ColDef("Lap", width=0.25),
            ColDef("Duration", width=0.5, textprops={
                   "ha": "right"}, border="l")
        ]

    # Different sizes depending on amound of data shown with filters
    size = (10, (results["Code"].size / 5.1) + 1)
    table = plot_table(results, col_defs, "Code", figsize=size)
    del results

    return table.figure, table.ax


def championship_table(data: list[dict], type: Literal["wcc", "wdc"]) -> tuple[Figure, Axes]:
    """Return matplotlib table displaying championship results."""

    # make dataframe from dict results
    df = pd.DataFrame(data)
    base_defs = [
        ColDef("Pos", width=0.35, textprops={"weight": "bold"}, border="r"),
        ColDef("Points", width=0.35, textprops={"ha": "right"}, border="l"),
        ColDef("Wins", width=0.35, textprops={"ha": "right"}, border="l"),
    ]

    # Driver
    if type == "wdc":
        size = (5, 10)
        col_defs = base_defs + \
            [ColDef("Driver", width=0.8, textprops={"ha": "left"})]
    # Constructors
    if type == "wcc":
        size = (6, 8)
        col_defs = base_defs + \
            [ColDef("Team", width=0.8, textprops={"ha": "left"})]

    table = plot_table(df, col_defs, "Pos", figsize=size)

    return table.figure, table.ax


def expand_events(incidents):
    """Expand multiple events in the same row into separate rows with the same lap."""

    expanded_rows = []
    for _, row in incidents.iterrows():
        statuses = row["TrackStatus"]
        events = utils.map_track_status(statuses).split(", ")

        for event in events:
            new_row = row.copy()
            new_row["Event"] = event
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df["Change"] = expanded_df["Event"] != expanded_df["Event"].shift(
        1)

    return expanded_df[expanded_df["Change"] == True].drop(columns=["Change"]).reset_index(drop=True)


def get_track_events(session: Session):
    """Return a DataFrame with lap number and event description, e.g. safety cars."""
    incidents = (
        Laps(session.laps.loc[:, ["LapNumber", "TrackStatus"]].dropna())
        .pick_track_status("123456789", how="any")
        .groupby("LapNumber").min()
        .reset_index()
    )
    # Map the status codes to names
    incidents["Event"] = incidents["TrackStatus"].apply(utils.map_track_status)
    result = expand_events(incidents)

    return result


def incidents_table(df: pd.DataFrame) -> tuple[Figure, Axes]:
    """Return table listing track retirements and status events."""
    df = df.rename(columns={"LapNumber": "Lap"})
    col_defs = [
        ColDef("Lap", width=0.15, textprops={"weight": "bold"}, border="r"),
        ColDef("Event", width=0.5, textprops={"ha": "right"})
    ]
    # Dynamic size
    size = (4, (df["Lap"].size / 3.333) + 1)
    table = plot_table(df, col_defs, "Lap", size)
    # Styling
    for cell in table.columns["Event"].cells:
        if cell.content in ("Safety Car", "Virtual Safety Car", "Yellow Flag(s)"):
            cell.text.set_color("#ffb300")
            cell.text.set_alpha(0.84)
        elif cell.content == "Red Flag":
            cell.text.set_color("#e53935")
            cell.text.set_alpha(0.84)
        elif cell.content == "Green Flag":
            cell.text.set_color("#43a047")
            cell.text.set_alpha(0.84)
        else:
            cell.text.set_color((1, 1, 1, 0.5))
    del df
    return table.figure, table.ax


def plot_race_schedule(data, year):

    df = pd.DataFrame(data)

    # Define column definitions for the table
    col_defs = [
        ColDef("Round", width=0.35, textprops={"weight": "bold"}, border="r"),
        ColDef("Country", width=0.8, textprops={"ha": "left"}, border="l"),
        ColDef("Circuit", width=1.0, textprops={"ha": "left"}),
        ColDef("Date", width=0.8, textprops={"ha": "left"})
    ]
    if year > 2020:
        df2 = df.drop(columns=["Event Format"], errors="ignore")
    else:
        df2 = df
    table = plot_table(df2, col_defs, "Round", figsize=(10, 8))
    if year > 2020:
        for i, row in df.iterrows():
            if row['Event Format'] != 'conventional':
                table.rows[i].set_hatch(
                    "//").set_facecolor("#b138dd").set_alpha(0.35)

    return table.figure, table.ax


def stints(data):
    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Define column definitions for the table
    col_defs = [

        ColDef("Driver", width=0.8, textprops={"ha": "left"}),
        ColDef("HARD", width=0.8, textprops={"ha": "left"}),
        ColDef("MEDIUM", width=0.8, textprops={"ha": "left"}),
        ColDef("SOFT", width=0.8, textprops={"ha": "left"})
    ]

    # Plot the table
    table = plot_table(df, col_defs, "Driver", figsize=(10, 8))

    return table.figure


def stints_driver(data):
    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Define column definitions for the table
    col_defs = [
        ColDef("Driver", width=0.8),
        ColDef("Stint", width=0.8),
        ColDef("Compound", width=0.8),
        ColDef("FreshTyre", width=0.8),
        ColDef("Lap", width=0.8),
        ColDef("TyreLife", width=0.8)
    ]

    # Plot the table
    table = plot_table(df, col_defs, "Driver", figsize=(6, 4))

    return table.figure


def racecontrol(messages, session):

    messages = pd.DataFrame(messages)
    messages.drop(['Category', 'Status', 'Flag', 'Scope',
                  'Sector', 'RacingNumber'], axis=1, inplace=True)
    max_per_file = 25
    messages['Time'] = messages['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    num_files = (len(messages) // max_per_file) + 1
    files = []
    if session == 'Race' or session == 'Sprint':
        messages = messages[['Time', 'Lap', 'Message']]
        col_defs = [
            ColDef("Time", width=0.2),
            ColDef("Lap", width=0.07),
            ColDef("Message", width=1.1, textprops={"ha": "left"})
        ]
        figsize = (20, 10)
    else:
        messages.drop(['Lap'], axis=1, inplace=True)
        col_defs = [

            ColDef("Time", width=0.2),
            ColDef("Message", width=1.6, textprops={"ha": "left"})
        ]
        figsize = (23, 13)

    for i in range(num_files):
        start_idx = i * max_per_file
        end_idx = min((i + 1) * max_per_file, len(messages))
        file_messages = messages.iloc[start_idx:end_idx]

        fig = plot_race_control_table(
            file_messages, col_defs, "Time", figsize=figsize)
        files.append(fig.figure)

    return files


def plot_chances(data):
    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Define column definitions for the table
    col_defs = [
        ColDef("Position", width=0.35, textprops={
               "weight": "bold"}, border="r"),
        ColDef("Driver", width=0.8, textprops={"ha": "left"}, border="l"),
        ColDef("Current Points", width=0.8, textprops={"ha": "left"}),
        ColDef("Theoretical max points", width=0.8, textprops={"ha": "left"}),
        ColDef("Can win?", width=0.8, textprops={"ha": "left"})
    ]

    if (df["Can win?"] == "Yes").sum() == 1:
        idx = df[df["Can win?"] == "Yes"].index[0]
        table = plot_table(df, col_defs, "Position", figsize=(10, 8))
        table.rows[idx].set_hatch(
            "//").set_facecolor("#FFD700").set_alpha(0.35)
    else:
        # Plot the table without any highlights
        table = plot_table(df, col_defs, "Position", figsize=(10, 8))

    return table.figure


def grid_table(data: list[dict]) -> tuple[Figure, Axes]:
    """Return table showing the season grid."""

    df = pd.DataFrame(data)
    col_defs = [
        ColDef("Code", width=0.4, textprops={"weight": "bold"}, border="r"),
        ColDef("No", width=0.35),
        ColDef("Name", width=0.9, textprops={"ha": "left"}),
        ColDef("Age", width=0.35),
        ColDef("Nationality", width=0.75, textprops={"ha": "left"}),
        ColDef("Team", width=0.8, textprops={"ha": "left"}),
    ]
    table = plot_table(df, col_defs, "Code", figsize=(10, 10))

    return table.figure, table.ax


def laptime_table(df: pd.DataFrame) -> tuple[Figure, Axes]:
    """Return table with fastest lap data per driver."""
    df = df.drop("Delta", axis=1)
    col_defs = [
        ColDef("Rank", width=0.25, textprops={"weight": "bold"}, border="r"),
        ColDef("Driver", width=0.25),
        ColDef("LapTime", width=0.35, title="Time", textprops={"ha": "right"}),
        ColDef("Lap", width=0.35),
        ColDef("Tyre", width=0.35),
        ColDef("ST", width=0.25)
    ]
    size = (6, (df["Driver"].size / 3.333) + 1)
    table = plot_table(df, col_defs, "Rank", figsize=size)
    table.rows[0].set_hatch("//").set_facecolor("#b138dd").set_alpha(0.35)
    table.columns["ST"].cells[df["ST"].idxmax()].text.set_color("#b138dd")
    del df

    return table.figure, table.ax


def sectors_table(df: pd.DataFrame) -> tuple[Figure, Axes]:
    """Return table with fastest sector times and speed."""
    sectors = df.loc[:, ["Driver", "S1", "S2", "S3", "ST"]]
    s_defs = [ColDef(c, width=0.4, textprops={
                     "ha": "right"}) for c in ("S1", "S2", "S3")]
    col_defs = [
        ColDef("Driver", width=0.25, textprops={"weight": "bold"}, border="r"),
        ColDef("ST", width=0.25, textprops={"ha": "right"}, border="l")
    ] + s_defs

    # Calculate table height based on rows
    size = (6, (df["Driver"].size / 3.333) + 1)

    table = plot_table(sectors, col_defs, "Driver", size)

    # Highlight fastest values
    table.columns["S1"].cells[df["Sector1Time"].idxmin()
                              ].text.set_color("#b138dd")
    table.columns["S2"].cells[df["Sector2Time"].idxmin()
                              ].text.set_color("#b138dd")
    table.columns["S3"].cells[df["Sector3Time"].idxmin()
                              ].text.set_color("#b138dd")
    table.columns["ST"].cells[df["ST"].idxmax()].text.set_color("#b138dd")
    del df

    return table.figure, table.ax


def get_top_role_color(member: discord.Member):
    try:
        # Sort roles by position, from highest to lowest
        roles = sorted(
            member.roles, key=lambda role: role.position, reverse=True)

        # Find the first role with a color other than the default color (which is usually 0)
        for role in roles:
            if role.color.value != 0:  # default color has a value of 0
                return role.color
        return discord.Color.default()
    except:
        return 0x2F3136
