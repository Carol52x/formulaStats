"Slash command parameter options."

from discord import Option
from f1.update import pitstop_year_list, fastf1_year_list, document_year_list, regulation_year_list


DriverOption = Option(
    str, default=None, description="Driver number, 3-letter code or surname")

RankedPitstopFilter = Option(
    str, choices=["Best", "Worst", "Ranked"],
    default="Ranked", description="Which stops to view (default ranked)")


class DriverOptionRequired(Option):
    def __init__(self, input_type=str, description="Driver number, 3-letter code or surname", **kwargs) -> None:
        super().__init__(input_type, description, **kwargs)


driveroption2 = Option(
    str,
    description="Enter Full Name of the driver."
)

driveroption2 = Option(
    str,
    description="Enter Full Name of the driver."
)
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
    choices=[True, False],
    description="default false",
    default=True)

DNFoption = Option(
    bool,
    choices=[True, False],
    description="Choose whether to include DNFs (default false)",
    default=False)

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
    choices=fastf1_year_list,
    description="The season year (default current)")

SeasonOption4 = Option(
    int,
    default=None,
    choices=pitstop_year_list,
    description="The season year (default current)")
SeasonOption5 = Option(
    int,
    default=None,
    choices=document_year_list,
    description="The season year (default current)")

SeasonOption6 = Option(
    int,
    default=None,
    choices=regulation_year_list,
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
