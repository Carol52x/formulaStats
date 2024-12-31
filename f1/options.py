"Slash command parameter options."

from discord import Option, AutocompleteContext
from f1.update import pitstop_year_list, fastf1_year_list, document_year_list, regulation_year_list, ergast_year_list
import fastf1
from f1.api import stats, ergast
from rapidfuzz import process
import requests
from bs4 import BeautifulSoup
import asyncio

url = 'https://en.wikipedia.org/wiki/List_of_Formula_One_drivers'
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
table = soup.find_all('table')[2]

driver_data = []

try:
    for row in table.find_all('tr')[1:]:
        columns = row.find_all('td')
        flags = row.find('img', {'class': 'mw-file-element'})
        if flags:
            nationality = flags['src']
        if columns:
            driver_dict = {
                'name': stats.parse_driver_name(columns[0].text.strip()),
            }
            driver_data.append(driver_dict)
except Exception as e:
    print(f"Error: {e}")

url_team = 'https://en.wikipedia.org/wiki/List_of_Formula_One_constructors'
response_team = requests.get(url_team)
soup_team = BeautifulSoup(response_team.text, "html.parser")
table_list = soup_team.find_all('table')
table_1 = table_list[1]
table_2 = table_list[2]
team_data = []

for row in table_1.find_all('tr')[1:]:
    columns = row.find_all('td')
    if columns:
        team_dict = {
            'constructor': stats.parse_brackets(columns[0].text.strip())
        }
        team_data.append(team_dict)

# Parse the second table
for row in table_2.find_all('tr')[1:]:
    columns = row.find_all('td')
    if columns:
        team_dict = {
            'constructor': stats.parse_brackets(columns[0].text.strip())
        }
        team_data.append(team_dict)

RankedPitstopFilter = Option(
    str, choices=["Best", "Worst", "Ranked"],
    default="Ranked", description="Which stops to view (default ranked)")


async def resolve_full_driver_name(ctx: AutocompleteContext):
    # Get the user input from ctx.value
    user_input = ctx.value

    # Extract the driver names from the driver_data list
    driver_names = [driver['name'] for driver in driver_data]

    # Perform fuzzy matching to find the best matches
    matches = process.extract(user_input, driver_names, limit=5)

    # Return the exact driver names for the top matches
    return [match[0] for match in matches]


async def resolve_team(ctx: AutocompleteContext):
    user_input = ctx.value
    team_names = [team['constructor'] for team in team_data]
    matches = process.extract(user_input, team_names, limit=5)
    return [match[0] for match in matches]


async def resolve_term(ctx: AutocompleteContext):
    user_input = ctx.value
    url = f"https://www.f1technical.net/glossary/{user_input[0].lower()}"
    html = await asyncio.to_thread(lambda: requests.get(url))
    soup = BeautifulSoup(html.content, 'html.parser')
    glossary = {}
    for dt in soup.find_all('dt'):
        dd = dt.find_next_sibling('dd')
        if dd:
            glossary[dt.get_text(strip=True)] = dd.get_text(strip=True)
    terms = [term for term in glossary.keys()]
    matches = process.extract(user_input, terms, limit=5)
    return [match[0] for match in matches]


async def resolve_years_ergast(ctx: AutocompleteContext):
    if 'year' in ctx.options:
        # Use the first 4 characters to match years
        year_prefix = str(ctx.value)[:3]
        year_list = [str(y) for y in ergast_year_list]
        # Filter the list of years based on the first 3 digits of user input
        # Return the years as integers
        return [int(y) for y in year_list if y.startswith(year_prefix)][:25]


async def resolve_years_fastf1(ctx: AutocompleteContext):
    # Handle Year Autocomplete
    if 'year' in ctx.options:
        # Use the first 4 characters to match years
        year_prefix = str(ctx.value)[:3]
        year_list = [str(y) for y in fastf1_year_list]
        # Filter the list of years based on the first 3 digits of user input
        # Return the years as integers
        return [int(y) for y in year_list if y.startswith(year_prefix)][:25]


async def resolve_years_pitstop(ctx: AutocompleteContext):
    # Handle Year Autocomplete
    if 'year' in ctx.options:
        # Use the first 4 characters to match years
        year_prefix = str(ctx.value)[:3]
        year_list = [str(y) for y in pitstop_year_list]
        # Filter the list of years based on the first 3 digits of user input
        # Return the years as integers
        return [int(y) for y in year_list if y.startswith(year_prefix)][:25]


async def resolve_years_regulation(ctx: AutocompleteContext):
    # Handle Year Autocomplete
    if 'year' in ctx.options:
        # Use the first 4 characters to match years
        year_prefix = str(ctx.value)[:3]
        year_list = [str(y) for y in regulation_year_list]
        # Filter the list of years based on the first 3 digits of user input
        # Return the years as integers
        return [int(y) for y in year_list if y.startswith(year_prefix)][:25]


async def resolve_years_document(ctx: AutocompleteContext):
    # Handle Year Autocomplete
    if 'year' in ctx.options:
        # Use the first 4 characters to match years
        year_prefix = str(ctx.value)[:3]
        year_list = [str(y) for y in document_year_list]
        # Filter the list of years based on the first 3 digits of user input
        # Return the years as integers
        return [int(y) for y in year_list if y.startswith(year_prefix)][:25]


async def resolve_rounds(ctx: AutocompleteContext):
    year = ctx.options.get('year')  # Fetch the selected year
    if 'round' in ctx.options and year:
        # Only show rounds if year is selected
        event_list = fastf1.get_event_schedule(int(year), include_testing=False)[
            'EventName'].to_list()

        # Sanitize user input and event names
        # Handle None and normalize case
        user_input = (ctx.value or "").strip().lower()
        return [
            event for event in event_list
            if user_input in event.lower()  # Case-insensitive match
        ][:25]


async def resolve_sessions_by_year(ctx: AutocompleteContext):
    year = int(ctx.options.get("year"))
    if year < 2003:
        sessions = ["Race"]
    elif year < 2021:
        sessions = ["Qualifying", "Race"]
    elif year < 2023:
        sessions = ["Qualifying", "Sprint", "Race"]
    elif year == 2023:
        sessions = ["Qualifying", "Sprint Shootout", "Sprint", "Race"]
    else:
        sessions = ["Sprint Qualifying", "Sprint", "Qualifying", "Race"]
    return sessions


async def resolve_sessions(ctx: AutocompleteContext):
    year = int(ctx.options.get("year"))  # Fetch the selected year
    selected_event = ctx.options.get('round')  # Fetch the selected round

    if "session" in ctx.options and selected_event:
        schedule = fastf1.get_event_schedule(int(year), include_testing=False)

        # Define session options based on the year
        if year < 2003:
            sessions = ["Race"]
        elif year < 2018:
            sessions = ["Qualifying", "Race"]
        elif year < 2021:
            sessions = ["Practice 1", "Practice 2",
                        "Practice 3", "Qualifying", "Race"]
        elif year == 2021 or year == 2022:
            is_sprint = schedule[schedule['EventName'] ==
                                 selected_event]['EventFormat'].iloc[0] == 'sprint'
            if is_sprint:
                sessions = ["Practice 1", "Qualifying",
                            "Practice 2", "Sprint", "Race"]
            else:
                sessions = ["Practice 1", "Practice 2",
                            "Practice 3", "Qualifying", "Race"]
        elif year == 2023:
            is_sprint = schedule[schedule['EventName'] ==
                                 selected_event]['EventFormat'].iloc[0] == 'sprint_shootout'
            if is_sprint:
                sessions = ["Practice 1", "Qualifying",
                            "Sprint Shootout", "Sprint", "Race"]
            else:
                sessions = ["Practice 1", "Practice 2",
                            "Practice 3", "Qualifying", "Race"]
        else:
            is_sprint = schedule[schedule['EventName'] ==
                                 selected_event]['EventFormat'].iloc[0] == 'sprint_qualifying'
            if is_sprint:
                sessions = ["Practice 1", "Sprint Qualifying",
                            "Sprint", "Qualifying", "Race"]
            else:
                sessions = ["Practice 1", "Practice 2",
                            "Practice 3", "Qualifying", "Race"]

        # Filter the sessions based on user input
        return [session for session in sessions if ctx.value in session][:25]


async def resolve_drivers(ctx: AutocompleteContext):
    year = int(ctx.options.get("year"))
    round = ctx.options.get('round')
    session = ctx.options.get('session', "Race")
    ev = await stats.to_event(year, round)
    if year > 2017:
        s = await stats.load_session(ev, session)
        driver_list = s.drivers
        driver_names = [
            s.get_driver(driver)['Abbreviation']
            for driver in driver_list
            if s.get_driver(driver)['ClassifiedPosition'] != "W"
        ]

        matches = [
            driver_name for driver_name in driver_names if ctx.value.lower() in driver_name.lower()
        ]
    else:
        driver_names = []
        driver_dict = await ergast.get_all_drivers(year, ev['RoundNumber'])
        for i in driver_dict:
            driver_names.append(i['code'])
        matches = [
            driver_name for driver_name in driver_names if ctx.value.lower() in driver_name.lower()
        ]

    return matches[:25]


async def resolve_tyres(ctx: AutocompleteContext):
    year = int(ctx.options.get("year"))
    if year == 2018:
        choices = [
            "SOFT",
            "MEDIUM",
            "HARD",
            "INTERMEDIATE",
            "WET",
            "SUPERHARD",
            'SUPERSOFT',
            "ULTRASOFT",
            'HYPERSOFT'
        ]
    else:
        choices = ["SOFT",
                   "MEDIUM",
                   "HARD",
                   "INTERMEDIATE",
                   "WET"]
    return choices


async def resolve_laps(ctx: AutocompleteContext):
    year = int(ctx.options.get("year"))
    round = ctx.options.get('round')
    session = ctx.options.get('session') or "Race"
    ev = await stats.to_event(year, round)
    s = await stats.load_session(ev, session, messages=True)
    max_lap = s.race_control_messages['Lap'].max()
    lap_list = list(range(1, max_lap + 1))
    matches = [str(lap) for lap in lap_list if str(lap).startswith(ctx.value)]
    return matches[:25]


RecordOption = Option(
    str,
    choices=["Drivers", 'Constructors',
             "Engines", "Tyres", "Races", "Misc. Driver records",  "Misc. Driver records (part 2)", "Sprint records"],
    description="Choose the type of record to view. (default drivers)",
    default="Drivers")

RegulationOption = Option(
    str,
    choices=["Sporting Regulations", 'Technical Regulations',
             "Financial Regulations", "Operational Regulations"],
    description="Choose the type of regulation to view. (default Sporting)",
    default="Sporting Regulations")


EphemeralOption = Option(
    bool,
    choices=[True, False]
)

DNFoption = Option(
    bool,
    choices=[True, False],
    description="Choose whether to include DNFs (default false)",
    default=False)


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

AvgMedianOption = Option(
    str,
    choices=[
        "Average",
        "Median"
    ],
    default="Average",
    description="Quali gaps in median or average over a season. (default average)")


async def resolve_sessions_by_year_quali(ctx: AutocompleteContext):
    year = int(ctx.options.get("year"))
    if year < 2023:
        sessions = ['Qualifying']
    elif year == 2023:
        sessions = ["Qualifying", "Sprint Shootout"]
    else:
        sessions = ["Qualifying", "Sprint Qualifying"]
    return sessions
