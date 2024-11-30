import threading
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
import datetime
from f1.errors import MissingDataError
from f1.config import Config
from f1.api import ergast, stats
from f1 import options, utils
import io
import matplotlib.patheffects as pe
import matplotlib.image as mpim
import pytz
from matplotlib.ticker import MaxNLocator
import math
from f1.api.stats import get_top_role_color
import matplotlib
from fastf1.ergast import Ergast
from windrose import WindroseAxes
from plotly.io import show
import plotly.express as px
from matplotlib.ticker import MultipleLocator
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import logging
import matplotlib.pyplot as plt
from matplotlib import colormaps
import discord
import fastf1.plotting
import asyncio
import numpy as np
import pandas as pd
import seaborn as sns
from discord.commands import ApplicationContext
from discord.ext import commands
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
import time
from matplotlib.figure import Figure
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme='fastf1')
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"
matplotlib.use('agg')

ioz = io.BytesIO()

request_delay = 0.5
curr_year = int(datetime.datetime.now().year)

logger = logging.getLogger("f1-bot")

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
            print(f"An error occurred: {e}")
            return None


# Example usage in a loop
client = ErgastClient()

# Set the DPI of the figure image output; discord preview seems sharper at higher value


def roundnumber(round=None, year=None):
    if round == None and year == None or round == None and year == int(datetime.datetime.now().year):

        schedule = fastf1.get_event_schedule(
            int(datetime.datetime.now().year), include_testing=False)

        for index, row in schedule.iterrows():

            if row["Session5Date"] < pd.Timestamp(datetime.date.today(), tzinfo=pytz.utc):
                number = row['RoundNumber']
                round = number
    if year == None:
        year = int(datetime.datetime.now().year)
    if year < int(datetime.datetime.now().year) and round == None:
        round = max(fastf1.get_event_schedule(
            int(year), include_testing=False)['RoundNumber'])
    return [round, year]


DPI = 300
current_date = datetime.date.today()


def sectors_func(yr, rc, sn, d1, d2, lap, event, session):
    d1 = d1[0:3].upper()
    d2 = d2[0:3].upper()

    lap1 = lap
    lap2 = lap1

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')

    # Explore the lap data
    session.laps

    if (d1 == None or d1 == ''):
        d1 = session.laps.pick_fastest()['Driver']

    if (d2 == None or d2 == ''):
        d2 = session.laps.pick_fastest()['Driver']

    driver_1 = d1
    driver_2 = d2

    color_1 = 'green'
    color_2 = 'red'
    # Find the laps
    laps_driver_1 = session.laps.pick_driver(driver_1)
    laps_driver_2 = session.laps.pick_driver(driver_2)

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

    telemetry_driver_1 = fastest_driver_1.get_telemetry()
    telemetry_driver_2 = fastest_driver_2.get_telemetry()

    # Get the gap (delta time) between driver 1 and driver 2
    delta_time, ref_tel, compare_tel = fastf1.utils.delta_time(
        fastest_driver_1, fastest_driver_2)

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

    # Convert the coordinates to points, and then concat them into segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fastest_driver_array = telemetry['Fastest_driver_int'].to_numpy().astype(
        float)

    # The segments we just created can now be colored according to the fastest driver in a minisector
    cmap = ListedColormap([color_1, color_2])
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
    plt.box(False)
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

    # sn = session.event.get_session_name(sn)

    plt.suptitle(f"{yr} {event['EventName']} {sn} - Fastest Sectors\n" +
                 d1 + " (" + lap1 + ") vs " + d2 + " (" + lap2 + ")", size=25)

    image = io.BytesIO()

    plt.savefig(image, format='png')

    image.seek(0)
    file = discord.File(image, filename="plot.png")
    image.close()
    plt.close()
    return file


def weather(year, location, session, event, race):

    race_name = race.event.EventName
    df = race.laps

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
    fig.suptitle('Weather Data & Track Evolution \n'+race_name, fontsize=30)

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
    image = io.BytesIO()
    fig.set_tight_layout(False)
    plt.savefig(image, format='png')

    image.seek(0)
    file = discord.File(image, filename="plot.png")  # R
    image.close()
    plt.close()
    return file


def cornering_func(yr, rc, sn, d1, d2, dist1, dist2, event, session):

    d1 = d1[0:3].upper()
    d2 = d2[0:3].upper()
    lap1 = ""
    lap2 = ""

    # Get the laps
    laps = session.laps

    if (d1 == None or d1 == ''):
        d1 = laps.pick_fastest()['Driver']

    if (d2 == None or d2 == ''):
        d2 = laps.pick_fastest()['Driver']

    # Setting parameters
    driver_1, driver_2 = d1, d2

    car_data = laps.pick_driver(
        driver_1).pick_fastest().get_car_data().add_distance()
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
    laps_driver_1 = laps.pick_driver(driver_1)
    laps_driver_2 = laps.pick_driver(driver_2)

    if (lap1 == None or lap1 == ''):
        telemetry_driver_1 = laps_driver_1.pick_fastest().get_car_data().add_distance()
    else:
        temp_laps1 = laps_driver_1[laps_driver_1['LapNumber'] == int(
            lap1)].iloc[0]
        telemetry_driver_1 = temp_laps1.get_car_data().add_distance()

    if (lap2 == None or lap2 == ''):
        telemetry_driver_2 = laps_driver_2.pick_fastest().get_car_data().add_distance()
    else:
        temp_laps2 = laps_driver_2[laps_driver_2['LapNumber'] == int(
            lap2)].iloc[0]
        telemetry_driver_2 = temp_laps2.get_car_data().add_distance()

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
        (telemetry_driver_1['Distance'] >= distance_max)
    ])

    avg_speed_driver_2 = np.mean(telemetry_driver_2['Speed'].loc[
        (telemetry_driver_2['Distance'] >= distance_min) &
        (telemetry_driver_2['Distance'] >= distance_max)
    ])

    if avg_speed_driver_1 > avg_speed_driver_2:
        speed_text = f"{driver_1} {round(avg_speed_driver_1 - avg_speed_driver_2,2)}km/h faster"
    else:
        speed_text = f"{driver_2} {round(avg_speed_driver_2 - avg_speed_driver_1,2)}km/h faster"

    ##############################
    #
    # Setting everything up
    #
    ##############################
    plt.rcParams["figure.figsize"] = [13, 4]
    plt.rcParams["figure.autolayout"] = True

    telemetry_colors = {
        'Full Throttle': 'green',
        'Cornering': 'grey',
        'Brake': 'red',
    }

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
    fig, ax = plt.subplots(2)

    ##############################
    #
    # Lineplot for speed
    #
    ##############################

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

    ##############################
    #
    # Horizontal barplot for telemetry
    #
    ##############################
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

    ##############################
    #
    # Styling of the plot
    #
    ##############################
    # Set x-label
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

    # sn = session.event.get_session_name(sn)

    plt.suptitle(f"{yr} {event['EventName']} {sn}\n" +
                 d1 + " (" + lap1 + ") vs " + d2 + " (" + lap2 + ")", size=20)
    image = io.BytesIO()

    plt.savefig(image, format='png')

    image.seek(0)
    file = discord.File(image, filename="plot.png")
    image.close()
    plt.close()
    return file


def positions_func(yr):
    ergast = Ergast()
    races = ergast.get_race_schedule(yr)  # Races in year 2022
    results = []
    if yr == curr_year:
        schedule = fastf1.get_event_schedule(yr, include_testing=False)
        last_index = None
        for index, row in schedule.iterrows():
            if row["Session5Date"] < pd.Timestamp(current_date, tzinfo=pytz.utc):
                last_index = index

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

    # Get results. Note that we use the round no. + 1, because the round no.
    # starts from one (1) instead of zero (0)

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

# Write the bytes object to a BytesIO buffer
    buffer = io.BytesIO()
    buffer.write(fig_bytes)

# Reset the buffer position to the beginning
    buffer.seek(0)

# Now, you can return the buffer as a file
    file = discord.File(buffer, filename="plot.png")  # R
    buffer.close()
    plt.close()

    return file


def time_func(yr, rc, sn, driver1, driver2, lap, event, session):
    drivers = [driver1[0:3].upper(), driver2[0:3].upper()]

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
    fig, ax = plt.subplots()

    plt.rcParams["figure.figsize"] = [7, 5]
    plt.rcParams["figure.autolayout"] = True

    fast = 0
    t = 0
    vCar = 0
    car_data = 0

    i = 0
    for z in drivers:
        style = fastf1.plotting.get_driver_style(identifier=drivers[i],
                                                 style=['color', 'linestyle'],
                                                 session=session)

    while (i < len(drivers)):
        if (lap == None or lap == ''):
            fast = session.laps.pick_driver(drivers[i]).pick_fastest()
        else:
            driver_laps = session.laps.pick_driver(drivers[i])
            fast = driver_laps[driver_laps['LapNumber'] == int(lap)].iloc[0]
        car_data = fast.get_car_data()
        t = car_data['Time']
        vCar = car_data['Speed']
        style = fastf1.plotting.get_driver_style(identifier=drivers[i],
                                                 style=['color', 'linestyle'],
                                                 session=session)
        try:
            ax.plot(t, vCar, **style, label=str(drivers[i]))
        except:
            ax.plot(t, vCar, color='grey', label=str(drivers[i]))
        i = i+1

    title = str(drivers[0])

    i = 0
    while (i < len(drivers)):
        if (i+1 < len(drivers)):
            title = title + " vs " + str(drivers[i+1])
        i += 1

    # sn = session.event.get_session_name(sn)

    if (lap == None or lap == ''):
        plt.suptitle("Fastest Lap Comparison\n" +
                     f"{yr} {event['EventName']} {sn}\n" + title)
    else:
        plt.suptitle("Lap " + str(lap) + " Comparison " +
                     f"{yr} {event['EventName']} {sn}\n" + title)

    plt.setp(ax.get_xticklabels(), fontsize=7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Speed [Km/h]')
    ax.legend()
    image = io.BytesIO()
    ax.grid(True, alpha=0.1)
    ax.minorticks_on()

    plt.savefig(image, format='png')

    image.seek(0)
    file = discord.File(image, filename="plot.png")
    image.close()
    plt.close()
    return file


def tel_func(yr, rc, sn, d1, d2, lap1, lap2, event, session):
    d1 = d1[0:3].upper()
    d2 = d2[0:3].upper()

    laps = session.laps

    if (d1 == None or d1 == ''):
        d1 = laps.pick_fastest()['Driver']

    if (d2 == None or d2 == ''):
        d2 = laps.pick_fastest()['Driver']

    drv1 = d1
    drv2 = d2

    first_driver = laps.pick_driver(drv1)
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

    second_driver = laps.pick_driver(drv2)
    second_driver_info = session.get_driver(drv2)

    if (lap1 == None or lap1 == ''):
        first_driver = laps.pick_driver(drv1).pick_fastest()
    else:
        driver_laps = session.laps.pick_driver(drv1)
        first_driver = driver_laps[driver_laps['LapNumber'] == int(
            lap1)].iloc[0]

    if (lap2 == None or lap2 == ''):
        second_driver = laps.pick_driver(drv2).pick_fastest()
    else:
        driver_laps = session.laps.pick_driver(drv2)
        second_driver = driver_laps[driver_laps['LapNumber'] == int(
            lap2)].iloc[0]

    first_car = first_driver.get_car_data().add_distance()
    second_car = second_driver.get_car_data().add_distance()

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
    fig, ax = plt.subplots(7, 1, figsize=(15, 10), gridspec_kw={
        'height_ratios': [3, 3, 3, 3, 3, 3, 3],  # Equal height for all subplots
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

    # sn = session.event.get_session_name(sn)

    # sn = session.event.get_session_name(sn)

    fig.suptitle(f"{yr} {event['EventName']}\n" +
                 drv1 + " (" + lap1 + ") vs " + drv2 + " (" + lap2 + ")", size=15)

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

    ax[6].set_ylabel(drv1 + " ahead | " + drv2 + " ahead")

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

    ax[0].set_ylabel("Speed [km/h]")
    ax[1].set_ylabel("RPM [#]")
    ax[2].set_ylabel("Gear [#]")
    ax[3].set_ylabel("Throttle [%]")
    ax[4].set_ylabel(f"Brake [%]\n {drv2} | {drv1}")
    ax[5].set_ylabel("DRS")

    ax[0].get_xaxis().set_ticklabels([])
    ax[1].get_xaxis().set_ticklabels([])
    ax[2].get_xaxis().set_ticklabels([])
    ax[3].get_xaxis().set_ticklabels([])
    ax[4].get_xaxis().set_ticklabels([])

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

    plt.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.05)

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

    image = io.BytesIO()
    plt.tight_layout()
    plt.savefig(image, format='png')
    image.seek(0)
    file = discord.File(image, filename="plot.png")
    image.close()
    plt.close()
    return file


def driver_func(yr):
    schedule = fastf1.get_event_schedule(yr, include_testing=False)
    if yr == int(datetime.datetime.now().year):

        last_index = None
        for index, row in schedule.iterrows():

            if row["Session5Date"] < pd.Timestamp(datetime.date.today(), tzinfo=pytz.utc):
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

    # Initate a loop through all the rounds
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

        # Append the current round to our fial dataframe
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

    # Disable the gridlines
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

    # Save the plot
    # plt.show()
    image = io.BytesIO()

    plt.savefig(image, bbox_inches='tight', format='png')
    image.seek(0)
    file = discord.File(image, filename="plot.png")
    image.close()
    plt.close()
    return file


def const_func(yr):
    schedule = fastf1.get_event_schedule(yr, include_testing=False)
    if yr == int(datetime.datetime.now().year):

        last_index = None
        for index, row in schedule.iterrows():

            if row["Session5Date"] < pd.Timestamp(datetime.date.today(), tzinfo=pytz.utc):
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

    # Initate a loop through all the rounds
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
    image = io.BytesIO()

    plt.savefig(image, bbox_inches='tight', format='png')
    image.seek(0)
    file = discord.File(image, filename="plot.png")
    image.close()
    plt.close()
    return file


def get_data2(year, session_type):
    current_date = datetime.date.today()
    # the code for this is kinda bad and complicated so imma leave comments for when i forget what any of this does
    team_list = {}
    color_list = {}
    check_list = {}

    team_fullName = {}
    driver_country = {}
    outstring = ''
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    last_index = None
    for index, row in schedule.iterrows():

        if row["Session5Date"] < pd.Timestamp(current_date, tzinfo=pytz.utc):
            number = row['RoundNumber']
            last_index = number

    for c in range(min(schedule.RoundNumber), last_index+1):
        # for c in range(min(schedule.index),min(schedule.index)+5):
        session = fastf1.get_session(
            year, schedule.loc[c, 'RoundNumber'], session_type)
        session.load(laps=False, telemetry=False,
                     weather=False, messages=False)
        results = session.results
        # print(results)
        # for driver in results.index:
        #     country_code = results.loc[driver,'CountryCode']
        #     if not (country_code == ''):
        #         driver_country.update({results.loc[driver,'Abbreviation']:country_code})

        # for each race, reset the boolean
        # boolean is used to prevent same driver being updated twice in the same race since it iterates through every driver
        # EX: if VER finished ahead of PER, first VER is updated to +1, boolean is set to true,
        # that way when loop gets to PER it doesnt add +1 to VER again since boolean is true
        # until the next race when they can be updated again
        for i in check_list.keys():
            check_list.update({i: False})

        for i in results['TeamId']:
            team_results = results.loc[lambda df: df['TeamId'] == i]
            if len(team_results.index) < 2:
                break
            # testing
            # if (i == "Ferrari"):
            #     print(team_results)
            # print(team_results[['Abbreviation','ClassifiedPosition','Status','TeamColor','TeamId']])

            # dictionary format:
            # teamName:
            #   driverPairing:
            #       driver: #ofRacesFinishedAheadofTeammate
            #       otherDriver: #ofRacesFinishedAheadofTeammate
            if (team_list.get(i) is None):
                team_fullName.update(
                    {i: team_results.loc[min(team_results.index), 'TeamName']})
                team_list.update({i: {}})
                color_list.update(
                    {i: team_results.loc[min(team_results.index), 'TeamColor']})

            drivers = []
            for j in team_results.index:
                drivers.append(team_results.loc[j, 'Abbreviation'])
            drivers = sorted(drivers)
            pairing = ''.join(drivers)

            if (team_list.get(i).get(pairing) is None):
                team_list.get(i).update({pairing: {}})

            for abbreviation in team_results['Abbreviation']:
                if team_list.get(i).get(pairing).get(abbreviation) is None:
                    team_list.get(i).get(pairing).update({abbreviation: 0})

            curr_abbr = team_results.loc[team_results.index[0], 'Abbreviation']

            # figure out which races to ignore
            both_drivers_finished = True
            if (session_type == 'Race'):
                dnf = ['D', 'E', 'W', 'F', 'N']
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
    return team_list, color_list, team_fullName  # , outstring


def get_data(year, session_type):
    team_list = {}
    color_list = {}
    check_list = {}
    team_fullName = {}
    driver_country = {}
    outstring = ''
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for c in range(min(schedule.RoundNumber), max(schedule.RoundNumber)+1):
        # for c in range(min(schedule.index),min(schedule.index)+5):
        session = fastf1.get_session(
            year, schedule.loc[c, 'RoundNumber'], session_type)
        session.load(laps=False, telemetry=False,
                     weather=False, messages=False)
        results = session.results
        for i in check_list.keys():
            check_list.update({i: False})

        for i in results['TeamId']:
            team_results = results.loc[lambda df: df['TeamId'] == i]
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
                drivers.append(team_results.loc[j, 'Abbreviation'])
            drivers = sorted(drivers)
            pairing = ''.join(drivers)

            if (team_list.get(i).get(pairing) is None):
                team_list.get(i).update({pairing: {}})

            for abbreviation in team_results['Abbreviation']:
                if team_list.get(i).get(pairing).get(abbreviation) is None:
                    team_list.get(i).get(pairing).update({abbreviation: 0})

            curr_abbr = team_results.loc[team_results.index[0], 'Abbreviation']

            # figure out which races to ignore
            both_drivers_finished = True
            if (session_type == 'Race'):
                dnf = ['D', 'E', 'W', 'F', 'N']
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
    return team_list, color_list, team_fullName  # , outstring


def make_plot(data, colors, year, session_type, team_names, filepath):
    plt.clf()
    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
    fig, ax = plt.subplots(1, figsize=(13, 9))

    fig.suptitle(f'{year} {session_type} Head to Head', size=20, y=0.95)
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
                color = "#ffffff"
            ax.barh(pairing, driver_wins, color=color,)  # edgecolor = 'black')

            ax.text(0, offset, team.replace("_", " ").title(), ha='center', fontsize=10,  path_effects=[
                pe.withStroke(linewidth=2, foreground="black")])

            # label the bars
            for i in range(len(drivers)):
                # Check if the driver participated
                if driver_wins[i] <= 0:
                    driver_name = drivers[i]
                    driver_names.append(driver_name)
                    wins_string = f'{-1*driver_wins[i]}'
                    ax.text(min(driver_wins[i] - 0.6, -1.2), offset - 0.2, wins_string,  fontsize=20,
                            horizontalalignment='right', path_effects=[pe.withStroke(linewidth=4, foreground="black")])
                else:
                    driver_name = drivers[i]
                    driver_names.append(driver_name)
                    wins_string = f'{driver_wins[i]}'
                    ax.text(driver_wins[i] + 0.6, offset - 0.2, wins_string,  fontsize=20,
                            horizontalalignment='left', path_effects=[pe.withStroke(linewidth=4, foreground="black")])
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

    # watermark

    ax.text(0.1, 1.03, '', transform=ax.transAxes,
            fontsize=13, ha='center')

    ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
    offset = 0
    # label drivers
    for i in range(len(driver_names)):
        if (i % 2) == 0:
            ax.text(xabs_max, offset-0.2, driver_names[i], fontsize=20, horizontalalignment='right', path_effects=[
                    pe.withStroke(linewidth=4, foreground="black")])
        else:
            ax.text(-xabs_max, math.floor(offset)-0.2, driver_names[i], fontsize=20, horizontalalignment='left', path_effects=[
                    pe.withStroke(linewidth=4, foreground="black")])
        offset += 0.5
    plt.rcParams['savefig.dpi'] = 300

    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def get_embed(year, session_type, ctx):
    image = io.BytesIO()
    data, colors, names = get_data(year, session_type)
    make_plot(data, colors, year, session_type, names, image)
    image.seek(0)
    file = discord.File(image, filename="image.png")
    top_role_color = get_top_role_color(ctx.author)
    title = f"Teammate {session_type} Head to Head {year}"
    image.close()
    return customEmbed(title=title, image_url='attachment://image.png', colour=top_role_color), file


def get_embed2(year, session_type, ctx):
    image = io.BytesIO()
    data, colors, names = get_data2(year, session_type)
    make_plot(data, colors, year, session_type, names, image)
    image.seek(0)
    file = discord.File(image, filename="image.png")
    title = f"Teammate {session_type} Head to Head {year}"
    top_role_color = get_top_role_color(ctx.author)
    image.close()
    return customEmbed(title=title, image_url='attachment://image.png', colour=top_role_color), file


def get_event_data2(session_type, year, category):
    current_date = datetime.date.today()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    first_index = schedule.index[0]
    last_index = None
    for index, row in schedule.iterrows():

        if row["Session5Date"] < pd.Timestamp(current_date, tzinfo=pytz.utc):
            number = row['RoundNumber']
            last_index = number
    first_round = schedule.loc[first_index, 'RoundNumber']
    last_round = last_index
    # print(first_index)
    # print(max_index)
    # print(first_round)
    # print(last_round)

    # event = fastf1.get_session(year,first_round,session_type)
    # event.load()

    driver_positions = {}
    driver_average = {}
    driver_colors = {}
    driver_racesParticipated = {}

    for i in range(first_round, last_round+1):
        event = fastf1.get_session(year, i, session_type)
        event.load(laps=False, telemetry=False, weather=False, messages=False)
        results = event.results
        results.dropna(subset=['Position'], inplace=True)

        # print(results)

        for driver in event.drivers:
            if (category == 'Drivers'):
                currDriver_abbreviation = results.loc[driver, 'Abbreviation']
            else:
                currDriver_abbreviation = results.loc[driver, 'TeamName']

            if driver_positions.get(currDriver_abbreviation) is None:
                driver_positions.update({currDriver_abbreviation: 0})

            if driver_racesParticipated.get(currDriver_abbreviation) is None:
                driver_racesParticipated.update({currDriver_abbreviation: 0})

            if session_type == 'Race':
                currDriver_position = results.loc[driver, 'ClassifiedPosition']
            else:
                currDriver_position = results.loc[driver, 'Position']

            currDriver_total = driver_positions.get(currDriver_abbreviation)

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
    # print(driver_positions)
    # print(driver_colors)
    # print(driver_racesParticipated)
    for key in driver_positions.keys():
        try:
            driver_average.update({key: driver_positions.get(
                key)/driver_racesParticipated.get(key)})
        except:
            print('div by 0')
    return driver_average, driver_colors


def get_event_data(session_type, year, category):
    schedule = fastf1.get_event_schedule(year=year, include_testing=False)
    first_index = schedule.index[0]
    max_index = max(schedule.RoundNumber)
    first_round = schedule.loc[first_index, 'RoundNumber']
    last_round = schedule.loc[max_index, 'RoundNumber']
    # print(first_index)
    # print(max_index)
    # print(first_round)
    # print(last_round)

    # event = fastf1.get_session(year,first_round,session_type)
    # event.load()

    driver_positions = {}
    driver_average = {}
    driver_colors = {}
    driver_racesParticipated = {}

    for i in range(first_round, last_round+1):
        event = fastf1.get_session(year, i, session_type)
        event.load(laps=False, telemetry=False, weather=False, messages=False)
        results = event.results
        results.dropna(subset=['Position'], inplace=True)
        # print(results)

        for driver in event.drivers:
            if (category == 'Drivers'):
                currDriver_abbreviation = results.loc[driver, 'Abbreviation']
            else:
                currDriver_abbreviation = results.loc[driver, 'TeamName']

            if driver_positions.get(currDriver_abbreviation) is None:
                driver_positions.update({currDriver_abbreviation: 0})

            if driver_racesParticipated.get(currDriver_abbreviation) is None:
                driver_racesParticipated.update({currDriver_abbreviation: 0})

            if session_type == 'Race':
                currDriver_position = results.loc[driver, 'ClassifiedPosition']
            else:
                currDriver_position = results.loc[driver, 'Position']

            currDriver_total = driver_positions.get(currDriver_abbreviation)

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
    # print(driver_positions)
    # print(driver_colors)
    # print(driver_racesParticipated)
    for key in driver_positions.keys():
        try:
            driver_average.update({key: driver_positions.get(
                key)/driver_racesParticipated.get(key)})
        except:
            print('div by 0')
    return driver_average, driver_colors


def plot_avg(driver_positions, driver_colors, session_type, year, category, filepath):

    plt.clf()
    driver_positions = dict(
        sorted(driver_positions.items(), key=lambda x: x[1]))

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
    # set directory for later use
    # create the bar plot and size
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
    ax.minorticks_off()

    # testing for possible median bars
    # plt.barh("VER", 5, color='none', edgecolor='blue', hatch="/", linewidth=1,alpha=0.6)
    annotated = False
    for driver in driver_positions.keys():
        curr_color = driver_colors.get(driver)
        # print(curr_color)
        # print(curr_color == 'nan')
        if ((curr_color != 'nan') and (curr_color != '')):
            plt.barh(driver, driver_positions.get(
                driver), color=f'#{curr_color}')
        else:
            if not annotated:
                plt.figtext(
                    0.91, 0.01, "*Some color data is unavailable", ha="center")
                annotated = True
            plt.barh(driver, driver_positions.get(driver), color='#ffffff')
    # add f1buddy pfp

    for i, position in enumerate(driver_positions.values()):
        ax.text(position + 0.1, i,
                f"   {str(round(position,2))}", va='center',  fontsize=20)

    plt.rcParams['savefig.dpi'] = 300
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def get_embed_and_image(year, session_type, category, ctx):

    pos, colors = get_event_data(
        session_type=session_type, year=year, category=category)

    # Create a BytesIO object
    image_bytes_io = io.BytesIO()

    plot_avg(pos, colors, session_type, year, category, image_bytes_io)

    # Reset the file position to the beginning
    image_bytes_io.seek(0)

    file = discord.File(image_bytes_io, filename="image.png")
    description = "DNS/DNF excluded from calculation"
    title = f"Average {category} {session_type} Finish Position {year}"
    top_role_color = get_top_role_color(ctx.author)
    image_bytes_io.close()
    return customEmbed(title=title, description=description, image_url='attachment://image.png', colour=top_role_color), file


def get_embed_and_image2(year, session_type, category, ctx):

    pos, colors = get_event_data2(
        session_type=session_type, year=curr_year, category=category)

    # Create a BytesIO object
    image_bytes_io = io.BytesIO()
    top_role_color = get_top_role_color(ctx.author)

    plot_avg(pos, colors, session_type,
             curr_year, category, image_bytes_io)

    # Reset the file position to the beginning
    image_bytes_io.seek(0)

    file = discord.File(image_bytes_io, filename="image.png")
    description = "DNS/DNF excluded from calculation"
    title = f"Average {category} {session_type} Finish Position {year}"
    image_bytes_io.close()
    return customEmbed(title=title, description=description, image_url='attachment://image.png', colour=top_role_color), file


class Plot(commands.Cog, guild_ids=Config().guilds):
    """Commands to create charts from race data."""

    def __init__(self, bot: discord.Bot):
        self.bot = bot

    fastf1.plotting.setup_mpl(misc_mpl_mods=False, mpl_timedelta_support=True, color_scheme='fastf1')
   

    @commands.slash_command(name="cornering", description="Cornering Comparison of any two drivers.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def cornering(self, ctx: ApplicationContext, driver1: discord.Option(str, required=True),
                        driver2: discord.Option(str, required=True), year: options.SeasonOption3, round: options.RoundOption, session: options.SessionOption,
                        distance1: discord.Option(int, default=2000), distance2: discord.Option(int, default=2000)):

        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True, telemetry=True)
        loop = asyncio.get_running_loop()
        file = await loop.run_in_executor(None, cornering_func, year, round, session, driver1, driver2, distance1, distance2, event, s)

        embed = discord.Embed(title=f'Cornering Analysis: {driver1[0:3].upper()} vs {driver2[0:3].upper()}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(name="speed-comparison", description="Speed Comparison (Time or Distance) of any two drivers.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def speed(self, ctx: ApplicationContext, driver1: discord.Option(str, required=True),
                    driver2: discord.Option(str, required=True), year: options.SeasonOption3, round: options.RoundOption, session: options.SessionOption,  lap: options.LapOption):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True, telemetry=True)
        loop = asyncio.get_running_loop()

        file = await loop.run_in_executor(None, time_func, year, round, session, driver1, driver2, lap, event, s)

        embed = discord.Embed(
            title=f'Speed Comparison (Time) {driver1[0:3].upper()} vs {driver2[0:3].upper()}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(name="track-evolution", description="Trackside weather and evolution data.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def wt(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption, session: options.SessionOption):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]

        event = await stats.to_event(year, round)
        race = await stats.load_session(event, session, weather=True, laps=True)

        await utils.check_season(ctx, year)
        loop = asyncio.get_running_loop()
        file = await loop.run_in_executor(None, weather, year, round, session, event, race)
        embed = discord.Embed(title=f'Track Evolution: {event.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(name="standings-heatmap", description="Plot WDC standings on a heatmap.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def heatmap(self, ctx: ApplicationContext, year: options.SeasonOption3):

        if year == None:
            year = int(datetime.datetime.now().year)

        await utils.check_season(ctx, year)
        loop = asyncio.get_running_loop()
        file = await loop.run_in_executor(None, positions_func, year)
        embed = discord.Embed(title=f"WDC standings (heatmap) {year}",
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(name="race-trace", description="Lap Comparison of participating drivers", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def racetrace(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True, telemetry=True)
        session.load()
        drivers = []
        dri = pd.unique(session.laps['Driver'])
        for a in dri:
            drivers.append(a)

        fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
        fig, ax = plt.subplots()

        plt.rcParams["figure.figsize"] = [14, 12]
        plt.rcParams["figure.autolayout"] = True

    #

        laps = session.laps
    # laps = laps.loc[laps['PitOutTime'].isna() & laps['PitInTime'].isna() & laps['LapTime'].notna()]
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

        avg = laps.groupby(['DriverNumber', 'Driver'])['LapTimeSeconds'].mean()

    # calculate the diff vs the best average. You could average the average if you want?
        laps['Difference'] = laps['LapTimeSeconds'] - avg.min()

        laps['Cumulative'] = laps.groupby('Driver')['Difference'].cumsum()

        fig, ax = plt.subplots()
        fig.set_size_inches(15, 7)

        for driver in drivers:
            temp = laps.loc[laps['Driver'] == driver][[
                'Driver', 'LapNumber', 'Cumulative']]
            if not temp.empty:
                style = fastf1.plotting.get_driver_style(identifier=temp.iloc[0]['Driver'],
                                                         style=[
                                                             'color', 'linestyle'],
                                                         session=session)

                ax.plot(temp['LapNumber'], temp['Cumulative'],
                        **style, label=temp.iloc[0]['Driver'])

        ax.set_xlabel('Lap Number')
        ax.set_ylabel('Cumulative gap (in seconds)')
        ax.set_title("Race Trace - " +
                     f"{session.event.year} {session.event['EventName']}\n")
        start, end = 0, ax.get_xlim()[1]
        ax.xaxis.set_ticks(np.arange(start, int(end), 10))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, pos: '{:0.0f}'.format(x)))

        ax.legend()
        ax.grid(which="minor", alpha=0.1)
        ax.minorticks_on()
        image = io.BytesIO()

        plt.savefig(image, format="png")
        image.seek(0)
        file = discord.File(image, filename="plot.png")
        embed = discord.Embed(title=f'Race Trace: {ev["EventName"]} ',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(name='standing-history', description="Standing History of either WDC or WCC", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def standing(self, ctx: ApplicationContext, year: options.SeasonOption2, category: options.category):

        if year == None:
            year = int(datetime.datetime.now().year)
        await utils.check_season(ctx, year)
        if category == 'Drivers':
            loop = asyncio.get_running_loop()
            file = await loop.run_in_executor(None, driver_func, year)
            embed = discord.Embed(title=f'WDC History: {year}',
                                  color=get_top_role_color(ctx.author))
            embed.set_image(url="attachment://plot.png")
            await ctx.respond(embed=embed, file=file)
        else:
            loop = asyncio.get_running_loop()
            file = await loop.run_in_executor(None, const_func, year)
            embed = discord.Embed(title=f'WCC History: {year}',
                                  color=get_top_role_color(ctx.author))
            embed.set_image(url="attachment://plot.png")
            await ctx.respond(embed=embed, file=file)

    @commands.slash_command(description="Compare fastest lap telemetry between two drivers.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def telemetry(self, ctx: ApplicationContext,
                        driver1: discord.Option(str, required=True),
                        driver2: discord.Option(str, required=True),
                        year: options.SeasonOption3, round: options.RoundOption,
                        session: options.SessionOption, lap1: options.LapOption1, lap2: options.LapOption2):
        """Plot lap telemetry (speed, distance, rpm, gears, brake) between two driver's fastest lap."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        event = await stats.to_event(year, round)
        session = await stats.load_session(event, session, laps=True, telemetry=True)

        loop = asyncio.get_running_loop()
        # await interaction.followup.send(content='h2h')
        f = await loop.run_in_executor(None, tel_func, year, round, session, driver1, driver2, lap1, lap2, event, session)

        embed = discord.Embed(
            title=f'Telemetry: {driver1[0:3].upper()} vs {driver2[0:3].upper()}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        embed.set_footer(
            text="Please note that the Brake traces are in binary and therfore are not an accurate representation of the actual telemetry.")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(name="h2h", description="Head to Head stats.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def h2hnew(self, ctx: ApplicationContext, year: options.SeasonOption2, session: options.SessionOption2):

        if year == None:
            year = int(datetime.datetime.now().year)

        if year == int(datetime.datetime.now().year):

            loop = asyncio.get_running_loop()

            dc_embed, file = await loop.run_in_executor(None, get_embed2, year, session, ctx)

            if not (file is None):
                await ctx.respond(embed=dc_embed.embed, file=file)
            else:
                await ctx.respond("Info not available!")
        else:
            loop = asyncio.get_running_loop()

            dc_embed, file = await loop.run_in_executor(None, get_embed, year, session, ctx)

            if not (file is None):
                await ctx.respond(embed=dc_embed.embed, file=file)
            else:
                await ctx.respond("Info not available!")

    @commands.slash_command(name="avgpos", description="Average position of a driver or a team in a span of season.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def positions(self, ctx: ApplicationContext, category: options.category, session: options.SessionOption2, year: options.SeasonOption2):

        if year == None:
            year = int(datetime.datetime.now().year)
        await utils.check_season(ctx, year)
        if year == int(datetime.datetime.now().year):
            loop = asyncio.get_running_loop()
            dc_embed, file = await loop.run_in_executor(None, get_embed_and_image2, year, session, category, ctx)
            if file != None:
                await ctx.respond(embed=dc_embed.embed, file=file)
            else:
                await ctx.respond(embed=dc_embed.embed)
        else:
            loop = asyncio.get_running_loop()
            dc_embed, file = await loop.run_in_executor(None, get_embed_and_image, year, session, category, ctx)
            if file != None:
                await ctx.respond(embed=dc_embed.embed, file=file)
            else:
                await ctx.respond(embed=dc_embed.embed)

    @commands.slash_command(description="Plot which gear is being used at which point of the track", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def gear(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        """Get a color coded Gear shift changes track mapping """

        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        # Load laps and telemetry data
        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True, telemetry=True)
        if not session.f1_api_support:
            raise MissingDataError("Lap data not available before 2018.")

        # Filter laps to the driver's fastest and get telemetry for the lap

        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry()
        x = np.array(tel['X'].values)
        y = np.array(tel['Y'].values)
        cmap = colormaps['Paired']
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        gear = tel['nGear'].to_numpy().astype(float)

# Create a figure and subplot
        fig, ax = plt.subplots(sharex=True, sharey=True)

# Create a LineCollection
        lc_comp = LineCollection(
            segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
        lc_comp.set_array(gear)
        lc_comp.set_linewidth(4)

# Add the LineCollection to the subplot
        ax.add_collection(lc_comp)

# Set axis properties
        ax.axis('equal')
        ax.tick_params(labelleft=False, left=False, labelbottom=False)

# Set title
        title = plt.suptitle(
            f"Fastest Lap Gear Shift Visualization\n"
            f"{lap['Driver']} - {session.event['EventName']} {session.event.year}"
        )

# Add colorbar
        cbar = plt.colorbar(mappable=lc_comp, label="Gear",
                            boundaries=np.arange(1, 10))
        cbar.set_ticks(np.arange(1.5, 9.5))
        cbar.set_ticklabels(np.arange(1, 9))

# Save the plot to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)
        file = discord.File(buffer, filename="plot.png")
        embed = discord.Embed(title=f'Gear Shift plot: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(name="tyre-strats", description="Tyre Strategies of the drivers' in a race.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def tyre_strats(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True, telemetry=True)
        session.load()
        if not session.f1_api_support:
            raise MissingDataError("Lap data not available before 2018.")
        laps = session.laps
        drivers = session.drivers
        drivers = [session.get_driver(driver)["Abbreviation"]
                   for driver in drivers]
        stints = laps[["Driver", "Stint",
                       "Compound", "LapNumber", "FreshTyre"]]
        stints = stints.groupby(["Driver", "Stint", "Compound", "FreshTyre"])
        stints = stints.count().reset_index()
        stints = stints.rename(columns={"LapNumber": "StintLength"})
        fig, ax = plt.subplots(figsize=(10, 10))

        for driver in drivers:
            driver_stints = stints.loc[stints["Driver"] == driver]

            previous_stint_end = 0
            for idx, row in driver_stints.iterrows():
                hatch = '' if row["FreshTyre"] else '/'
                plt.barh(
                    y=driver,
                    width=row["StintLength"],
                    left=previous_stint_end,
                    color=fastf1.plotting.get_compound_color(
                        row["Compound"], session),
                    edgecolor="black",
                    hatch=hatch
                )
                previous_stint_end += row["StintLength"]

        plt.title(
            f"{session.event['EventName']} {session.event.year} Tyre Strats")
        plt.xlabel("Lap Number")
        plt.grid(False)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        file = discord.File(buffer, filename="plot.png")
        embed = discord.Embed(title=f'Tyre Strategies: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        embed.set_footer(
            text='The stripes (if any) represents that the tyre is a used set.', icon_url=None)

        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(description="Plot driver position changes in the race.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def positionchanges(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        """Line graph per driver showing position for each lap."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        # Load the data
        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True)

        # Check API support
        if not session.f1_api_support:
            raise MissingDataError("Lap data not available before 2018.")

        fig = Figure(figsize=(8.5, 5.46), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        # Plot the drivers position per lap
        for d in session.drivers:
            laps = session.laps.pick_drivers(d)
            id = laps["Driver"].iloc[0]
            style = fastf1.plotting.get_driver_style(identifier=id,
                                                     style=['color', 'linestyle'],
                                                     session=session)
            ax.plot(laps["LapNumber"], laps["Position"], label=id,
                    **style)

        # Presentation
        ax.set_title(
            f"Race Position - {ev['EventName']} ({ev['EventDate'].year})")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Position")
        ax.set_yticks(np.arange(1, len(session.drivers) + 1))
        ax.tick_params(axis="y", right=True, left=True,
                       labelleft=True, labelright=False)
        ax.invert_yaxis()
        ax.legend(bbox_to_anchor=(1.01, 1.0))

        # Create image
        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Driver position changes: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(description="Show a bar chart comparing fastest laps in the session.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def fastestlaps(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption,
                          session: options.SessionOption):
        """Bar chart for each driver's fastest lap in `session`."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, session, laps=True)

        # Get the fastest lap per driver
        fastest_laps = stats.fastest_laps(s)
        # Filter out race start incidents
        if stats.get_session_type(session) == "R":
            fastest_laps = fastest_laps.loc[fastest_laps["Lap"] > 5]
        top = fastest_laps.iloc[0]

        # Map each driver to their team colour
        clr = [utils.get_driver_or_team_color(d, s, api_only=True)
               for d in fastest_laps["Driver"].values]

        # Plotting
        fig = Figure(figsize=(8, 6.75), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()
        bars = ax.barh(fastest_laps["Driver"],
                       fastest_laps["Delta"], color=clr)

        # Place a label next to each bar showing the delta in seconds

        bar_labels = [
            f"{d.total_seconds():.3f}" for d in fastest_laps["Delta"]]
        bar_labels[0] = ""
        ax.bar_label(bars,
                     labels=bar_labels,
                     label_type="edge",
                     fmt="%.3f",
                     padding=5,
                     fontsize=8)
        # Adjust xaxis to fit
        ax.set_xlim(
            right=fastest_laps["Delta"].max() + pd.Timedelta(seconds=0.5))

        ax.invert_yaxis()
        ax.grid(True, which="major", axis="x", zorder=0, alpha=0.2)
        ax.set_xlabel("Time Delta")
        ax.set_title(f"{s.name} - {ev['EventName']} ({ev['EventDate'].year})")
        fig.suptitle(f"Fastest: {top['LapTime']} ({top['Driver']})")

        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Fastest Laps: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(name="team-pace-delta", description="Rank team’s race pace from the fastest to the slowest.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def team_pace(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        race = await stats.load_session(ev, "R", laps=True, telemetry=True)
        race.load()
        if not race.f1_api_support:
            raise MissingDataError("Lap data not available before 2018.")
        laps = race.laps.pick_quicklaps()

        transformed_laps = laps.copy()
        transformed_laps.loc[:,
                             "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

# order the team from the fastest (lowest median lap time) tp slower
        team_order = (
            transformed_laps[["Team", "LapTime (s)"]]
            .groupby("Team")
            .median()["LapTime (s)"]
            .sort_values()
            .index
        )


# make a color palette associating team names to hex codes
        team_palette = {team: fastf1.plotting.get_team_color(
            team, race) for team in team_order}
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(
            data=transformed_laps,
            x="Team",
            y="LapTime (s)",
            hue="Team",
            order=team_order,
            palette=team_palette,
            whiskerprops=dict(color="white"),
            boxprops=dict(edgecolor="white"),
            medianprops=dict(color="grey"),
            capprops=dict(color="white"),
        )

        plt.title(f"{ev['EventName']} {year}")
        plt.grid(visible=True)

# x-label is redundant
        ax.set(xlabel=None)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)
        file = discord.File(buffer, filename="plot.png")
        embed = discord.Embed(title=f'Team Pace delta: {ev.EventName} ',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(name="driver-lap-time-distribution", description="View driver(s) laptime distribution on track.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def driver_laps(self, ctx: ApplicationContext, driver: options.DriverOptionRequired(),
                          year: options.SeasonOption3, round: options.RoundOption):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        if driver is None:
            raise ValueError("Specify a driver.")

        # Load laps and telemetry data
        ev = await stats.to_event(year, round)
        race = await stats.load_session(ev, "R", laps=True, telemetry=True)
        driver_laps = race.laps.pick_driver(
            driver[0:3].upper()).pick_quicklaps().reset_index()
        fig, ax = plt.subplots(figsize=(8, 8))

        sns.scatterplot(data=driver_laps,
                        x="LapNumber",
                        y="LapTime",
                        ax=ax,
                        hue="Compound",
                        palette=fastf1.plotting.get_compound_mapping(race),
                        s=80,
                        linewidth=0,
                        legend='auto')
        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Lap Time")

# The y-axis increases from bottom to top by default
# Since we are plotting time, it makes sense to invert the axis
        ax.invert_yaxis()
        plt.suptitle(
            f"{driver.upper()} Laptimes in the {year} {ev['EventName']}")

# Turn on major grid lines
        plt.grid(color='w', which='major', axis='both')
        sns.despine(left=True, bottom=True)

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)
        file = discord.File(buffer, filename="plot.png")
        embed = discord.Embed(
            title=f'Driver lap time distribution: {driver[0:3].upper()}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(name="track-speed", description="View driver speed on track.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def track_speed(self, ctx: ApplicationContext, driver: options.DriverOptionRequired(),
                          year: options.SeasonOption3, round: options.RoundOption):
        """Get the `driver` fastest lap data and use the lap position and speed
        telemetry to produce a track visualisation.
        """
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        if driver is None:
            raise ValueError("Specify a driver.")

        # Load laps and telemetry data
        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True, telemetry=True)

        # Filter laps to the driver's fastest and gettitle='', telemetry for the lap
        drv_id = utils.find_driver(driver, await ergast.get_all_drivers(year, ev["RoundNumber"]))["code"]
        lap = session.laps.pick_drivers(drv_id).pick_fastest()
        pos = lap.get_pos_data()
        car = lap.get_car_data()
        circuit_info = session.get_circuit_info()
        angle = circuit_info.rotation / 180 * np.pi
        # Reshape positional data to 3-d array of [X, Y] segments on track
        # (num of samples) x (sample row) x (x and y pos)
        # Then stack the points to get the beginning and end of each segment so they can be coloured
        rotated_pos_x = pos["X"] * np.cos(angle) - pos["Y"] * np.sin(angle)
        rotated_pos_y = pos["X"] * np.sin(angle) + pos["Y"] * np.cos(angle)
        points = np.array([rotated_pos_x, rotated_pos_y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        speed = car["Speed"]
        del lap, car

        fig = Figure(figsize=(12, 8), dpi=DPI, layout="constrained")
        ax = fig.subplots(sharex=True, sharey=True)
        ax.axis("off")

        # Plot the track and map segments to colors
        ax.plot(rotated_pos_x, rotated_pos_y, color="black", linestyle="-", linewidth=8, zorder=0)
        ax.axis('equal')

        norm = Normalize(speed.min(), speed.max())
        lc = LineCollection(segs, cmap="plasma", norm=norm, linestyle="-", linewidth=4)
        lc.set_array(speed)
        speed_line = ax.add_collection(lc)

        # Add the color bar at a better position
        cax = fig.add_axes([0.25, 0.02, 0.5, 0.025])  # Adjusted for more spacing
        fig.colorbar(speed_line, cax=cax, location="bottom", label="Speed (km/h)")

        fig.suptitle(f"{drv_id} Track Speed - {ev['EventDate'].year} {ev['EventName']}", size=16)

        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Speed visualisation: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(name="track-sectors", description="Compare fastest driver sectors on track map.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def track_sectors(self, ctx: ApplicationContext, first: options.DriverOptionRequired(),
                            second: options.DriverOptionRequired(), year: options.SeasonOption3,
                            round: options.RoundOption, session: options.SessionOption,
                            lap: options.LapOption):
        """Plot a track map showing where a driver was faster based on minisectors."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True, telemetry=True)
        loop = asyncio.get_running_loop()
        f = await loop.run_in_executor(None, sectors_func, year, round, session, first, second, lap, event, s)

        # Check API support

        embed = discord.Embed(title=f'Fastest Sectors comparison: {event.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(description="Show the position gains/losses per driver in the race.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def gains(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        """Plot each driver position change from starting grid position to finish position as a bar chart."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        # Load session results data
        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R")
        data = stats.pos_change(s)

        fig = Figure(figsize=(10, 5), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        # Plot pos diff for each driver
        for row in data.itertuples():
            bar = ax.bar(
                x=row.Driver,
                height=row.Diff,
                color="firebrick" if int(row.Diff) < 0 else "forestgreen",
                label=row.Diff,
            )
            ax.bar_label(bar, label_type="center")
        del data

        ax.set_title(
            f"Pos Gain/Loss - {ev['EventName']} ({(ev['EventDate'].year)})")
        ax.set_xlabel("Driver")
        ax.set_ylabel("Change")
        ax.grid(True, alpha=0.1)

        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(
            title=f'Driver position gains/losses: {ev.EventName}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(name="tyre-choice", description="Percentage distribution of tyre compounds.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def tyre_choice(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption,
                          session: options.SessionOption):
        """Plot the distribution of tyre compound for all laps in the session."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        # Get lap data and count occurance of each compound
        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, session, laps=True)
        t_count = s.laps["Compound"].value_counts()

        # Calculate percentages and sort
        t_percent = t_count / len(s.laps) * 100
        sorted_count = t_count.sort_values(ascending=False)
        sorted_percent = t_percent.loc[sorted_count.index]

        # Get tyre colours
        clrs = [fastf1.plotting.get_compound_color(
            i, s) for i in sorted_count.index]

        fig = Figure(figsize=(8, 6), dpi=DPI, layout="constrained")
        ax = fig.add_subplot(aspect="equal")

        ax.pie(sorted_percent, colors=clrs, autopct="%1.1f%%",
               textprops={"color": "black"})

        ax.legend(sorted_count.index)
        ax.set_title(
            f"Tyre Distribution - {session}\n{ev['EventName']} ({ev['EventDate'].year})")

        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Tyre choices: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(name="lap-compare", description="Compare laptime difference between two drivers.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def compare_laps(self, ctx: ApplicationContext,
                           first: options.DriverOptionRequired(),
                           second: options.DriverOptionRequired(),
                           year: options.SeasonOption3, round: options.RoundOption):
        """Plot the lap times between two drivers for all laps, excluding pitstops and slow laps."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R", laps=True, telemetry=True)
        # Get driver codes from the identifiers given
        drivers = [utils.find_driver(d, await ergast.get_all_drivers(year, ev["RoundNumber"]))["code"]
                   for d in (first, second)]

        # Group laps using only quicklaps to exclude pitstops and slow laps
        laps = s.laps.pick_drivers(drivers).pick_quicklaps()
        times = laps.loc[:, ["Driver", "LapNumber",
                             "LapTime"]].groupby("Driver")
        del laps

        fig = Figure(figsize=(8, 5), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        style1 = fastf1.plotting.get_driver_style(identifier=drivers[0],
                                                  style=['color', 'linestyle'],
                                                  session=s)
        style2 = fastf1.plotting.get_driver_style(identifier=drivers[1],
                                                  style=['color', 'linestyle'],
                                                  session=s)
        for d, t in times:
            style = style1 if d == drivers[0] else style2
            ax.plot(
                t["LapNumber"],
                t["LapTime"], **style,
                label=d

            )
        del times

        ax.set_title(
            f"Lap Difference -\n{ev['EventName']} ({ev['EventDate'].year})")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Time")
        ax.grid(which="minor", alpha=0.1)
        ax.minorticks_on()
        ax.legend()

        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(
            title=f'Laptime Comparison between two drivers: {first[0:3].upper()} vs {second[0:3].upper()}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(name="lap-distribution",
                            description="Violin plot comparing distribution of laptimes on different tyres.", integration_types={
                                discord.IntegrationType.guild_install,
                                discord.IntegrationType.user_install,
                            })
    async def lap_distribution(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        """Plot a swarmplot and violin plot showing laptime distributions and tyre compound
        for the top 10 point finishers."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R", laps=True)

        # Check API support
        if not s.f1_api_support:
            raise MissingDataError(
                "Session does not support lap data before 2018.")

        # Get the point finishers
        point_finishers = s.drivers[:10]

        laps = s.laps.pick_drivers(
            point_finishers).pick_quicklaps().set_index("Driver")
        # Convert laptimes to seconds for seaborn compatibility
        laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
        labels = [s.get_driver(d)["Abbreviation"] for d in point_finishers]
        compounds = laps["Compound"].unique()

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.violinplot(data=laps,
                       x=laps.index,
                       y="LapTime (s)",
                       inner=None,
                       scale="area",
                       order=labels,
                       palette=[utils.get_driver_or_team_color(d, s) for d in labels])

        sns.swarmplot(data=laps,
                      x="Driver",
                      y="LapTime (s)",
                      order=labels,
                      hue="Compound",
                      palette=[fastf1.plotting.get_compound_color(
                          c, s) for c in compounds],
                      linewidth=0,
                      size=5)
        del laps

        ax.set_xlabel("Driver (Point Finishers)")
        ax.set_title(
            f"Lap Distribution - {ev['EventName']} ({ev['EventDate'].year})")

        sns.despine(left=True, right=True)

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)
        file = discord.File(buffer, filename="plot.png")
# Now send the plot image as part of a message
        embed = discord.Embed(
            title=f'Lap distribution on violin plot: {ev.EventName}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file)

    @commands.slash_command(name="tyre-performance",
                            description="Plot the performance of each tyre compound based on the age of the tyre.", integration_types={
                                discord.IntegrationType.guild_install,
                                discord.IntegrationType.user_install,
                            })
    async def tyreperf(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        """Plot a line graph showing the performance of each tyre compound based on the age of the tyre."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R", laps=True)

        data = stats.tyre_performance(s)
        compounds = data["Compound"].unique()

        fig = Figure(figsize=(10, 5), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        for cmp in compounds:
            mask = data["Compound"] == cmp
            ax.plot(
                data.loc[mask, "TyreLife"].values,
                data.loc[mask, "Seconds"].values,
                label=cmp,
                color=fastf1.plotting.get_compound_color(cmp, s)
            )
        del data

        ax.set_xlabel("Tyre Life")
        ax.set_ylabel("Lap Time (s)")
        ax.set_title(
            f"Tyre Performance - {ev['EventDate'].year} {ev['EventName']}")
        ax.legend()
        ax.grid(which="minor", alpha=0.1)
        ax.minorticks_on()
        ax.grid(True, alpha=0.1)
        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Tyre degradation: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(description="Plots the delta in seconds between two drivers over a lap.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def gap(self, ctx: ApplicationContext, driver1: options.DriverOptionRequired(),
                  driver2: options.DriverOptionRequired(), year: options.SeasonOption3,
                  round: options.RoundOption, session: options.SessionOption, lap: options.LapOption):
        """Get the delta over lap distance between two drivers and return a line plot.

        `driver1` is comparison, `driver2` is reference lap.
        """
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, session, laps=True, telemetry=True)
        yr, rd = ev["EventDate"].year, ev["EventName"]

        # Check lap data support
        if not s.f1_api_support:
            raise MissingDataError("Lap data not available for the session.")

        # Check lap number is valid and within range
        if lap and int(lap) > s.laps["LapNumber"].unique().max():
            raise ValueError("Lap number out of range.")

        # Get drivers
        drivers = [utils.find_driver(d, await ergast.get_all_drivers(year, ev["RoundNumber"]))["code"]
                   for d in (driver1, driver2)]

        # Load each driver lap telemetry
        telemetry = {}
        for d in drivers:
            try:
                if lap:
                    driver_lap = s.laps.pick_drivers(
                        d).pick_laps(int(lap)).iloc[0]
                else:
                    driver_lap = s.laps.pick_drivers(d).pick_fastest()
                telemetry[d] = driver_lap.get_car_data(
                    interpolate_edges=True).add_distance()
            except Exception:
                raise MissingDataError(f"Cannot get telemetry for {d}.")

        # Get interpolated delta between drivers
        # where driver2 is ref lap and driver1 is compared
        delta = stats.compare_lap_telemetry_delta(
            telemetry[drivers[1]], telemetry[drivers[0]])

        # Mask the delta values to plot + green and - red
        ahead = np.ma.masked_where(delta >= 0., delta)
        behind = np.ma.masked_where(delta < 0., delta)

        fig = Figure(figsize=(10, 3), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        lap_label = f"Lap {lap}" if lap else "Fastest Lap"
        # Use ref lap distance for X
        x = telemetry[drivers[1]]["Distance"].values
        ax.plot(x, ahead, color="green")
        ax.plot(x, behind, color="red")
        ax.axhline(0.0, linestyle="--", linewidth=0.5,
                   color="w", zorder=0, alpha=0.5)
        ax.set_title(
            f"{drivers[0]} Delta to {drivers[1]} ({lap_label})\n{yr} {rd} | {session}").set_fontsize(16)
        ax.set_ylabel(
            f"<-  {drivers[0]}  |  {drivers[1]}  ->\n gap in seconds")
        ax.set_xlabel('Track Distance in meters')
        ax.grid(True, alpha=0.1)

        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Driver lap delta: {driver1[0:3].upper()} vs {driver2[0:3].upper()}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)

    @commands.slash_command(name="avg-lap-delta",
                            description="Bar chart comparing average time per driver with overall race average as a delta.", integration_types={
                                discord.IntegrationType.guild_install,
                                discord.IntegrationType.user_install,
                            })
    async def avg_lap_delta(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        """Get the overall average lap time of the session and plot the delta for each driver."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R", laps=True)
        yr, rd = ev["EventDate"].year, ev["EventName"]

        # Check lap data support
        if not s.f1_api_support:
            raise MissingDataError("Lap data not available for the session.")

        # Get the overall session average
        session_avg: pd.Timedelta = s.laps.pick_wo_box()["LapTime"].mean()

        fig = Figure(figsize=(10, 6), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        # Plot the average lap delta to session average for each driver
        for d in s.drivers:
            laps = (s.laps
                    .pick_drivers(d)
                    .pick_wo_box()
                    .pick_laps(range(5, s.total_laps + 1))
                    .loc[:, ["Driver", "LapTime"]])
            driver_avg: pd.Timedelta = laps["LapTime"].mean()

            # Filter out non-runners
            if pd.isna(driver_avg):
                continue

            delta = session_avg.total_seconds() - driver_avg.total_seconds()
            driver_id = laps["Driver"].iloc[0]
            ax.bar(x=driver_id, height=delta, width=0.75,
                   color=utils.get_driver_or_team_color(driver_id, s, api_only=True))
            del laps

        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="y", which="minor", grid_alpha=0.1)
        ax.tick_params(axis="y", which="major", grid_alpha=0.3)
        ax.grid(True, which="both", axis="y")
        ax.set_xlabel("Finishing Order")
        ax.set_ylabel("Delta (s)")
        ax.set_title(
            f"{yr} {rd}\nDelta to Average ({utils.format_timedelta(session_avg)})").set_fontsize(16)

        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Average lap delta: {ev.EventName} ',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f)


def setup(bot: discord.Bot):
    bot.add_cog(Plot(bot))
