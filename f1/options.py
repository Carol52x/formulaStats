import fastf1
"Slash command parameter options."

from discord import Option
from datetime import datetime
from datetime import date
import pandas as pd
import pytz
current_date = date.today()
current_year = int(datetime.now().year)
schedule = fastf1.get_event_schedule(
    int(datetime.now().year), include_testing=False)
first_index = schedule.index[0]
last_index = None

for index, row in schedule.iterrows():

    if row["Session5Date"] < pd.Timestamp(date.today(), tzinfo=pytz.utc):
        number = row['RoundNumber']
        last_index = number
first_round = schedule.loc[first_index, 'RoundNumber']
last_round = last_index
RankedPitstopFilter = Option(
    str, choices=["Best", "Worst", "Ranked"],
    default="Ranked", description="Which stops to view (default ranked)")

DriverOption = Option(
    str, default=None, description="Driver number, 3-letter code or surname")


class DriverOptionRequired(Option):
    def __init__(self, input_type=str, description="Driver number, 3-letter code or surname", **kwargs) -> None:
        super().__init__(input_type, description, **kwargs)


driveroption2 = Option(
    str,
    description="Enter Full Name of the driver."
)


EphemeralOption = Option(
    bool,
    choices=[True, False],
    default=True)


SectorFilter = Option(
    str,
    choices=["Time", "Speed"],
    description="The type of data to show",
    required=True
)

LaptimeFilter = Option(
    str,
    choices=["Fastest", "Slowest", "Top 5", "Bottom 5", "Ranked"],
    default="Ranked")

SeasonOption = Option(
    int,
    default=None,
    description="The season year (default current)")

SeasonOption2 = Option(
    int,
    default=None,
    description="The season year (default current)")
SeasonOption3 = Option(
    int,
    default=None,
    choices=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
    description="The season year (default current)")
RoundOption = Option(
    str,
    default=None,
    description="The race name, location or round number (default last race)")

TyreOption = Option(
    str,
    description="Choice of tyre compound (default none)",
    choices=[
        "SOFT",
        "MEDIUM",
        "HARD",
        "INTERMEDIATE",
        "WET"
    ],
    default=None
)

quizoption = Option(
    str,
    description="Choice of tyre compound (default none)",

    choices=[
        "1️⃣", "2️⃣", "3️⃣", "4️⃣"
    ],

)
category = Option(
    str,
    choices=[
        "Teams",
        "Drivers"
    ],
    default="Drivers",
    description="Average position of Teams or Drivers. (default drivers)")


SessionOption2 = Option(
    str,
    choices=[
        "Qualifying",
        "Race",
        "Sprint",
        "Sprint Qualifying/Sprint Shootout"
    ],
    default="Race",
    description="The session to view (default race)")


SessionOption = Option(
    str,
    choices=[
        "Practice 1",
        "Practice 2",
        "Practice 3",
        "Qualifying",
        "Sprint",
        "Sprint Qualifying",
        "Sprint Shootout",
        "Race"
    ],
    default="Race",
    description="The session to view (default race)")

LapOption = Option(
    int,
    min_value=1,
    default=None,
    description="Filter by lap number (optional, default fastest)"
)
LapOption1 = Option(
    int,
    min_value=1,
    default=None,
    description="lap number corresponding to driver 1 (default fastest)"
)
LapOption2 = Option(
    int,
    min_value=1,
    default=None,
    description="lap number corresponding to driver 2 (default fastest)"
)
