import datetime
from f1.api.stats import get_ephemeral_setting
from f1.errors import MissingDataError
from f1.config import Config
from f1.api import ergast, stats
from f1 import options, utils
from f1.api.stats import get_top_role_color
import matplotlib
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
from matplotlib.colors import Normalize
from f1.api.stats import roundnumber, sectors_func, tel_func, driver_func, const_func, heatmap_func, cornering_func, weather, h2h, averageposition
from matplotlib.figure import Figure
fastf1.plotting.setup_mpl(mpl_timedelta_support=True,
                          misc_mpl_mods=False, color_scheme='fastf1')
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"
matplotlib.use('agg')

logger = logging.getLogger("f1-bot")

DPI = 300


class Plot(commands.Cog, guild_ids=Config().guilds):
    """Commands to create charts from race data."""

    def __init__(self, bot: discord.Bot):
        self.bot = bot

    fastf1.plotting.setup_mpl(
        misc_mpl_mods=False, mpl_timedelta_support=True, color_scheme='fastf1')

    @commands.slash_command(name="cornering", description="Cornering Comparison of any two drivers.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def cornering(self, ctx: ApplicationContext, driver1: discord.Option(str, required=True),
                        driver2: discord.Option(str, required=True), year: options.SeasonOption3, round: options.RoundOption, session: options.SessionOption,
                        lap1: options.LapOption1, lap2: options.LapOption2,
                        distance1: discord.Option(int, default=2000), distance2: discord.Option(int, default=2000)):

        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True, telemetry=True)
        if lap1 and int(lap1) > s.laps["LapNumber"].unique().max():
            raise ValueError("Lap number out of range.")
        if lap2 and int(lap2) > s.laps["LapNumber"].unique().max():
            raise ValueError("Lap number out of range.")
        drivers = [utils.find_driver(d, await ergast.get_all_drivers(year, event["RoundNumber"]))["code"]
                   for d in (driver1, driver2)]
        driver1, driver2 = drivers[0], drivers[1]
        file = await cornering_func(year, round, session, driver1, driver2, lap1, lap2, distance1, distance2, event, s)

        embed = discord.Embed(title=f'Cornering Analysis: {driver1[0:3].upper()} vs {driver2[0:3].upper()}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

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
        file = await weather(year, round, session, event, race)
        embed = discord.Embed(title=f'Track Evolution: {event.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="standings-heatmap", description="Plot WDC standings on a heatmap.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def heatmap(self, ctx: ApplicationContext, year: options.SeasonOption3):

        if year == None:
            year = int(datetime.datetime.now().year)

        await utils.check_season(ctx, year)
        loop = asyncio.get_running_loop()
        file = await loop.run_in_executor(None, heatmap_func, year)
        embed = discord.Embed(title=f"WDC standings (heatmap) {year}",
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

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

        laps = await asyncio.to_thread(lambda: session.laps)
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
        file = file = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Race Trace: {ev["EventName"]} ',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

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
            await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))
        else:
            loop = asyncio.get_running_loop()
            file = await loop.run_in_executor(None, const_func, year)
            embed = discord.Embed(title=f'WCC History: {year}',
                                  color=get_top_role_color(ctx.author))
            embed.set_image(url="attachment://plot.png")
            await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(description="Compare fastest lap telemetry between two drivers.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def telemetry(self, ctx: ApplicationContext,
                        driver1: discord.Option(str, required=True),
                        driver2: discord.Option(str, required=True),
                        year: options.SeasonOption3, round: options.RoundOption,
                        session: options.SessionOption, lap1: options.LapOption1, lap2: options.LapOption2):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True, telemetry=True)
        drivers = [utils.find_driver(d, await ergast.get_all_drivers(year, event["RoundNumber"]))["code"]
                   for d in (driver1, driver2)]

        if lap1 and int(lap1) > await asyncio.to_thread(lambda: s.laps["LapNumber"].unique().max()):
            raise ValueError("Lap number out of range.")
        if lap2 and int(lap2) > await asyncio.to_thread(lambda: s.laps["LapNumber"].unique().max()):
            raise ValueError("Lap number out of range.")
        driver1, driver2 = drivers[0], drivers[1]
        f = await tel_func(year, round, session, driver1, driver2, lap1, lap2, event, s)

        embed = discord.Embed(
            title=f'Telemetry: {driver1[0:3].upper()} vs {driver2[0:3].upper()}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        embed.set_footer(
            text="Please note that the Brake traces are in binary and therfore are not an accurate representation of the actual telemetry.\nThere are also inherent inaccuracies with the way lap delta is calculated but at the moment there is no better way to calculate the said delta.")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="h2h", description="Head to Head stats.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def h2hnew(self, ctx: ApplicationContext, year: options.SeasonOption2, session: options.SessionOption2, include_dnfs: options.DNFoption):

        if year == None:
            year = int(datetime.datetime.now().year)
        if session == "Sprint Qualifying/Sprint Shootout":
            if year == 2023:
                session = "Sprint Shootout"
            else:
                session = "Sprint Qualifying"
        await utils.check_season(ctx, year)
        try:
            dc_embed, file = await h2h(year, session, ctx, include_dnfs)
            await ctx.respond(embed=dc_embed.embed, file=file, ephemeral=get_ephemeral_setting(ctx))
        except:
            await ctx.respond("Data for older seasons are very limited!", ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="avgpos", description="Average position of a driver or a team in a span of season.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def positions(self, ctx: ApplicationContext, category: options.category, session: options.SessionOption2, year: options.SeasonOption2, include_dnfs: options.DNFoption):

        if year == None:
            year = int(datetime.datetime.now().year)
        if session == "Sprint Qualifying/Sprint Shootout":
            if year == 2023:
                session = "Sprint Shootout"
            else:
                session = "Sprint Qualifying"
        await utils.check_season(ctx, year)
        try:
            dc_embed, file = await averageposition(session, year, category, ctx, include_dnfs)
            await ctx.respond(embed=dc_embed.embed, file=file, ephemeral=get_ephemeral_setting(ctx))
        except:
            await ctx.respond("Data unavailable.", ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(description="Plot which gear is being used at which point of the track", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def gear(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        """Get a color coded Gear shift changes track mapping """
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]

        # Load laps and telemetry data
        await utils.check_season(ctx, year)
        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True, telemetry=True)
        if not session.f1_api_support:
            raise MissingDataError("Lap data not available before 2018.")

        # Get circuit info and rotation angle
        circuit_info = session.get_circuit_info()
        track_angle = circuit_info.rotation / 180 * np.pi

        def rotate(xy, *, angle):
            rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                                [-np.sin(angle), np.cos(angle)]])
            return np.matmul(xy, rot_mat)

        # Filter laps to the driver's fastest and get telemetry for the lap
        lap = await asyncio.to_thread(lambda: session.laps.pick_fastest())
        tel = await asyncio.to_thread(lambda: lap.get_telemetry())
        x = np.array(tel['X'].values)
        y = np.array(tel['Y'].values)

        # Rotate the track coordinates
        coords = np.column_stack((x, y))  # Combine x and y into a single array
        rotated_coords = await asyncio.to_thread(rotate, coords, angle=track_angle)
        x_rotated, y_rotated = rotated_coords[:, 0], rotated_coords[:, 1]

        # Prepare for visualization
        cmap = colormaps['Paired']
        points = np.array([x_rotated, y_rotated]).T.reshape(-1, 1, 2)
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

        # Save the plot to a file
        file = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Gear Shift plot: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="tyre-strats", description="Tyre Strategies of the drivers' in a race.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def tyre_strats(self, ctx: ApplicationContext, year: options.SeasonOption3, round: options.RoundOption):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True, telemetry=True)
        session.load()
        if not session.f1_api_support:
            raise MissingDataError("Lap data not available before 2018.")
        laps = await asyncio.to_thread(lambda: session.laps)
        drivers = session.drivers
        drivers = [session.get_driver(driver)["Abbreviation"]
                   for driver in drivers]
        stints = laps[["Driver", "Stint",
                       "Compound", "LapNumber", "FreshTyre"]]
        stints = stints.groupby(["Driver", "Stint", "Compound", "FreshTyre"])
        stints = stints.count().reset_index()
        stints = stints.rename(columns={"LapNumber": "StintLength"})
        stints = stints[stints["Compound"] != "UNKNOWN"]
        fig, ax = plt.subplots(figsize=(10, 10))
        added_compounds = set()
        for driver in drivers:
            driver_stints = stints.loc[stints["Driver"] == driver]

            previous_stint_end = 0
            for idx, row in driver_stints.iterrows():
                compound = row["Compound"]
        # Skip adding the compound to the legend if it has already been added
                if compound not in added_compounds:
                    added_compounds.add(compound)
                    label = compound  # Set the label for the first occurrence
                else:
                    label = ""
                hatch = '' if row["FreshTyre"] else '/'
                plt.barh(
                    y=driver,
                    width=row["StintLength"],
                    left=previous_stint_end,
                    color=fastf1.plotting.get_compound_color(
                        row["Compound"], session),
                    edgecolor="black",
                    hatch=hatch,
                    label=label
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
        plt.legend(title="Compounds")
        plt.tight_layout()
        file = utils.plot_to_file(plt.gcf(), "plot")
        embed = discord.Embed(title=f'Tyre Strategies: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        embed.set_footer(
            text='The stripes (if any) represents that the tyre is a used set.', icon_url=None)

        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

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
            laps = await asyncio.to_thread(lambda: session.laps.pick_drivers(d))
            id = laps["Driver"].iloc[0]
            style = fastf1.plotting.get_driver_style(identifier=id,
                                                     style=[
                                                         'color', 'linestyle'],
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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

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
        fastest_laps = await stats.fastest_laps(s)
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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

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
        if not race.f1_api_support:
            raise MissingDataError("Lap data not available before 2018.")
        laps = await asyncio.to_thread(lambda: race.laps.pick_quicklaps())

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

        file = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Team Pace delta: {ev.EventName} ',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

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
        driver_laps = await asyncio.to_thread(lambda: race.laps.pick_drivers(
            driver[0:3].upper()).pick_quicklaps().reset_index())
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
        file = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(
            title=f'Driver lap time distribution: {driver[0:3].upper()}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

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
        lap = await asyncio.to_thread(lambda: session.laps.pick_drivers(drv_id).pick_fastest())
        pos = await asyncio.to_thread(lambda: lap.get_pos_data())
        car = await asyncio.to_thread(lambda: lap.get_car_data())
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

        fig, ax = plt.subplots(sharex=True, sharey=True)

        # Plot the track and map segments to colors
        ax.plot(rotated_pos_x, rotated_pos_y, color="black",
                linestyle="-", linewidth=8, zorder=0)
        ax.axis('equal')
        ax.tick_params(labelleft=False, left=False, labelbottom=False)

        norm = Normalize(speed.min(), speed.max())
        lc = LineCollection(segs, cmap="plasma", norm=norm,
                            linestyle="-", linewidth=4)
        lc.set_array(speed)
        speed_line = ax.add_collection(lc)

        # Adjusted for more spacing

        plt.colorbar(lc, label="Speed (km/h)")

        fig.suptitle(
            f"{drv_id} Track Speed - {ev['EventDate'].year} {ev['EventName']}", size=16)

        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Speed visualisation: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

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
        if lap and int(lap) > s.laps["LapNumber"].unique().max():
            raise ValueError("Lap number out of range.")
        loop = asyncio.get_running_loop()
        f = await sectors_func(year, round, session, first, second, lap, event, s)

        # Check API support

        embed = discord.Embed(title=f'Fastest Sectors comparison: {event.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

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
        data = await asyncio.to_thread(lambda: stats.pos_change(s))

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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

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
        stints = await asyncio.to_thread(lambda: s.laps)
        stints = stints[stints["Compound"] != "UNKNOWN"]
        t_count = stints["Compound"].value_counts()

        # Calculate percentages and sort
        t_percent = t_count / len(stints) * 100
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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

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
        laps = await asyncio.to_thread(lambda: s.laps.pick_drivers(drivers).pick_quicklaps())
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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

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
        point_finishers = await asyncio.to_thread(lambda: s.drivers[:10])

        laps = await asyncio.to_thread(lambda: s.laps.pick_drivers(
            point_finishers).pick_quicklaps().set_index("Driver"))
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
        file = utils.plot_to_file(fig, "plot")
# Now send the plot image as part of a message
        embed = discord.Embed(
            title=f'Lap distribution on violin plot: {ev.EventName}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

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

        data = await asyncio.to_thread(lambda: stats.tyre_performance(s))
        compounds = data["Compound"].unique()

        fig = Figure(figsize=(10, 5), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        for cmp in compounds:
            mask = data["Compound"] == cmp
            tyre_life = data.loc[mask, "TyreLife"].values
            lap_times = data.loc[mask, "Seconds"].values

            # Convert lap times to timedelta
            lap_times_formatted = [datetime.timedelta(seconds=sec) for sec in lap_times]

            ax.plot(
                tyre_life,
                lap_times,
                label=cmp,
                color=fastf1.plotting.get_compound_color(cmp, s),
            )

        del data

        # Format the y-axis ticks to display minutes and seconds
        def format_seconds(seconds):
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02}:{secs:06.3f}"

        # Format the y-axis ticks to display minutes and seconds
        def format_func(value, _):
            return format_seconds(value)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

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
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

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
        driver_laps = {}
        for d in drivers:
            try:
                if lap:
                    driver_lap = asyncio.to_thread(lambda: s.laps.pick_drivers(
                        d).pick_laps(int(lap))).iloc[0]
                else:
                    driver_lap = await asyncio.to_thread(lambda: s.laps.pick_drivers(d).pick_fastest())

                telemetry[d] = await asyncio.to_thread(lambda: driver_lap.get_car_data(
                    interpolate_edges=True).add_distance())
                driver_laps[d] = driver_lap
            except Exception:
                raise MissingDataError(f"Cannot get telemetry for {d}.")

        # Get interpolated delta between drivers
        # where driver2 is ref lap and driver1 is compared
        delta = fastf1.utils.delta_time(
            driver_laps[drivers[1]], driver_laps[drivers[0]])[0]

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
        embed.set_footer(
            text="Warning: This plot is actually not very accurate which is an inherent problem from the way this is calculated currently (There may not be a better way though.)")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="avg-lap-delta",
                            description="Bar chart comparing average time per driver with overall race average as a delta.",
                            integration_types={discord.IntegrationType.guild_install, discord.IntegrationType.user_install})
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

        # Get the overall session average lap time
        session_avg = await asyncio.to_thread(lambda: s.laps.pick_wo_box()["LapTime"].mean())

        fig = Figure(figsize=(10, 6), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        # Filter the laps for valid drivers
        valid_laps = s.laps.pick_wo_box().pick_laps(range(5, s.total_laps + 1))

        # Group by driver and calculate average lap time for each driver
        driver_avg_laps = valid_laps.groupby("Driver")["LapTime"].mean()

        # Calculate the delta for each driver to the session average and plot
        for driver_id, driver_avg in driver_avg_laps.items():
            if pd.isna(driver_avg):
                continue  # Skip non-runners

            delta = session_avg.total_seconds() - driver_avg.total_seconds()
            ax.bar(x=driver_id, height=delta, width=0.75,
                   color=utils.get_driver_or_team_color(driver_id, s, api_only=True))

        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="y", which="minor", grid_alpha=0.1)
        ax.tick_params(axis="y", which="major", grid_alpha=0.3)
        ax.grid(True, which="both", axis="y")
        ax.set_xlabel("Finishing Order")
        ax.set_ylabel("Delta (s)")
        ax.set_title(f"{yr} {rd}\nDelta to Average ({utils.format_timedelta(session_avg)})").set_fontsize(16)

        # Save the plot to a file
        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Average lap delta: {ev.EventName} ',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))


def setup(bot: discord.Bot):
    bot.add_cog(Plot(bot))
