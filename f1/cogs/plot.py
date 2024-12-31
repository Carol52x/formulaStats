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
import re
from matplotlib import colormaps
import discord
import fastf1.plotting
import asyncio
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from discord.commands import ApplicationContext
from discord.ext import commands
from f1.options import resolve_years_ergast, resolve_years_fastf1, resolve_drivers, resolve_laps, resolve_sessions, resolve_sessions_by_year, resolve_rounds
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
    async def cornering(self, ctx: ApplicationContext,  year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                        round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds),
                        session: discord.Option(str, "Select the session",  autocomplete=resolve_sessions),
                        driver1:  discord.Option(str, "Select driver 1", autocomplete=resolve_drivers),
                        driver2:  discord.Option(str, "Select driver 2", autocomplete=resolve_drivers),
                        lap1: discord.Option(int, "Select lap for driver 1 (default fastest)", default=None, autocomplete=resolve_laps),
                        lap2: discord.Option(int, "Select lap for driver 2 (default fastest)", default=None, autocomplete=resolve_laps)):

        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        session = stats.convert_shootout_to_qualifying(year, session)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True, telemetry=True)
        if lap1 and int(lap1) > s.laps["LapNumber"].unique().max():
            raise ValueError("Lap number out of range.")
        if lap2 and int(lap2) > s.laps["LapNumber"].unique().max():
            raise ValueError("Lap number out of range.")
        drivers = [utils.find_driver(d, await ergast.get_all_drivers(year, event["RoundNumber"]))["code"]
                   for d in (driver1, driver2)]
        driver1, driver2 = drivers[0], drivers[1]
        try:
            file = await cornering_func(year, round, session, driver1, driver2, lap1, lap2, "", "", event, s)
        except KeyError:
            await ctx.respond("No data for the driver found.\n-# *Why? The driver might not have participated in the given session.*")
            return

        embed = discord.Embed(title=f'Cornering Analysis: {driver1} vs {driver2}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")

        embed.description = '-# Public telemetry data, in general, is never extremely accurate. Check `/info` for details.'
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="track-evolution", description="Trackside weather and evolution data.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def wt(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                 round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds),
                 session: discord.Option(str, "Select the session",  autocomplete=resolve_sessions)):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        session = stats.convert_shootout_to_qualifying(year, session)

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
    async def heatmap(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1)):

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
    async def racetrace(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                        round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True, telemetry=True)
        drivers = []
        dri = pd.unique(session.laps['Driver'])
        for a in dri:
            drivers.append(a)

        fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
        fig, ax = plt.subplots()

        plt.rcParams["figure.figsize"] = [14, 12]
        plt.rcParams["figure.autolayout"] = True

        laps = await asyncio.to_thread(lambda: session.laps)
        sc_laps, vsc_laps = stats.find_sc_laps(laps)
        red_laps = stats.find_red_laps(laps)

        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        laps = laps.dropna(subset=['LapTimeSeconds'])

        avg = laps.groupby(['DriverNumber', 'Driver'])['LapTimeSeconds'].mean()

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
            else:
                continue

        ax.set_xlabel('Lap Number')
        ax.set_ylabel('Cumulative gap (in seconds)')
        ax.set_title("Race Trace - " +
                     f"{session.event.year} {session.event['EventName']}\n")
        stats.shade_sc_periods(sc_laps, vsc_laps, ax)
        stats.shade_red_flag(red_laps, ax)
        start, end = 0, ax.get_xlim()[1]
        ax.xaxis.set_ticks(np.arange(start, int(end), 10))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, pos: '{:0.0f}'.format(x)))

        ax.legend(bbox_to_anchor=(1.01, 1.0))
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
    async def standing(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_ergast),
                       category: options.category):

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
                        year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                        round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds),
                        session: discord.Option(str, "Select the session",  autocomplete=resolve_sessions),
                        driver1:  discord.Option(str, "Select driver 1", autocomplete=resolve_drivers),
                        driver2:  discord.Option(str, "Select driver 2", autocomplete=resolve_drivers),
                        lap1: discord.Option(int, "Select lap for driver 1 (default fastest)", default=None, autocomplete=resolve_laps),
                        lap2: discord.Option(int, "Select lap for driver 2 (default fastest)", default=None, autocomplete=resolve_laps)):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        session = stats.convert_shootout_to_qualifying(year, session)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True, telemetry=True)
        drivers = [utils.find_driver(d, await ergast.get_all_drivers(year, event["RoundNumber"]))["code"]
                   for d in (driver1, driver2)]

        if lap1 and int(lap1) > await asyncio.to_thread(lambda: s.laps["LapNumber"].unique().max()):
            raise ValueError("Lap number out of range.")
        if lap2 and int(lap2) > await asyncio.to_thread(lambda: s.laps["LapNumber"].unique().max()):
            raise ValueError("Lap number out of range.")
        driver1, driver2 = drivers[0], drivers[1]
        try:
            f = await tel_func(year, round, session, driver1, driver2, lap1, lap2, event, s)
        except KeyError:
            await ctx.respond("No data for the driver found.\n-# *Why? The driver might not have participated in the given session.*")
            return

        embed = discord.Embed(
            title=f'Telemetry: {driver1} vs {driver2}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        embed.description = "-# Please note that the Brake traces are in binary.\n-# Public telemetry data, in general, is never extremely accurate. Check `/info` for details."
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="h2h", description="Head to Head stats.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def h2hnew(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_ergast),
                     session: discord.Option(str, "Select the session",  autocomplete=resolve_sessions_by_year),
                     include_dnfs: options.DNFoption):
        try:
            dc_embed, file = await h2h(year, session, ctx, include_dnfs)
            await ctx.respond(embed=dc_embed.embed, file=file, ephemeral=get_ephemeral_setting(ctx))
        except:
            await ctx.respond("Data for older seasons are very limited!", ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="avgpos", description="Average position of a driver or a team in a span of season.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def positions(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_ergast),
                        session: discord.Option(str, "Select the session",  autocomplete=resolve_sessions_by_year), category: options.category, include_dnfs: options.DNFoption):
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
    async def gear(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                   round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
        """Get a color coded Gear shift changes track mapping """
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]

        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True, telemetry=True)

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
    async def tyre_strats(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                          round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
        try:
            round = roundnumber(round, year)[0]
            year = roundnumber(round, year)[1]
            await utils.check_season(ctx, year)
            ev = await stats.to_event(year, round)
            session = await stats.load_session(ev, "R", laps=True, telemetry=True)
            laps = await asyncio.to_thread(lambda: session.laps)
            drivers = session.drivers
            drivers = [session.get_driver(driver)["Abbreviation"]
                       for driver in drivers]
            stints = laps[["Driver", "Stint",
                           "Compound", "LapNumber", "FreshTyre"]]
            stints = stints.groupby(
                ["Driver", "Stint", "Compound", "FreshTyre"])
            stints = stints.count().reset_index()
            stints = stints.rename(columns={"LapNumber": "StintLength"})
            stints = stints[stints["Compound"] != "UNKNOWN"]
            sc_laps, vsc_laps = stats.find_sc_laps(laps)
            red_laps = stats.find_red_laps(laps)
            fig, ax = plt.subplots(figsize=(10, 10))
            added_compounds = set()
            legend_handles = {}
            if year == 2018:
                compound_label_mapping = {"SOFT": " ", "MEDIUM": " ", "HARD": " ", "HYPERSOFT": " ",
                                          "ULTRASOFT": " ",
                                          "SUPERSOFT": " ", "SUPERHARD": " "}
            else:
                absolute_compounds = await stats.get_compound_async(year, ev.EventName)
                compound_numbers = [int(s[1:]) for s in absolute_compounds]
                absolute_number_mapping = {i: j for i, j in zip(
                    compound_numbers, absolute_compounds)}
                soft_compound = absolute_number_mapping.get(
                    max(compound_numbers))
                hard_compound = absolute_number_mapping.get(
                    min(compound_numbers))
                remaining_compound = next(compound for compound, number in absolute_number_mapping.items()
                                          if number != soft_compound and number != hard_compound)
                compound_label_mapping = {
                    "SOFT": soft_compound,
                    "MEDIUM": absolute_number_mapping.get(remaining_compound),
                    "HARD": hard_compound
                }

            for driver in drivers:
                driver_stints = stints.loc[stints["Driver"] == driver]

                previous_stint_end = 0
                for idx, row in driver_stints.iterrows():
                    compound = row["Compound"]
                    absolute_compound = compound_label_mapping.get(
                        compound, compound)

                    if compound not in added_compounds:
                        if absolute_compound == " " or absolute_compound == compound:
                            label = f"{compound}"
                            added_compounds.add(compound)
                        else:
                            label = f"{compound} ({absolute_compound})"
                            added_compounds.add(compound)
                    else:
                        label = ""
                    if label and compound not in legend_handles:
                        legend_handles[compound] = Patch(
                            color=fastf1.plotting.get_compound_color(
                                compound, session),
                            label=label
                        )

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
            stats.shade_sc_periods(sc_laps, vsc_laps, ax)
            stats.shade_red_flag(red_laps, ax)
            if sc_laps.size > 0:
                legend_handles['Safety Car'] = Patch(
                    color='orange', alpha=0.5, label="Safety Car")
            if vsc_laps.size > 0:
                legend_handles['Virtual Safety Car'] = Patch(
                    color='yellow', alpha=0.5, label="Virtual Safety Car")
            if red_laps.size > 0:
                legend_handles['Red Flag'] = Patch(
                    color='red', alpha=0.5, label="Red Flag")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.legend(handles=list(legend_handles.values()), title="Legend")
            ax.grid(which="minor", alpha=0.1)
            ax.minorticks_on()
            plt.tight_layout()
            file = utils.plot_to_file(plt.gcf(), "plot")
            embed = discord.Embed(title=f'Tyre Strategies: {ev.EventName}',
                                  color=get_top_role_color(ctx.author))
            embed.set_image(url="attachment://plot.png")
            embed.description = '-# The stripes (if any) represents that the tyre is a used set.'

            await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))
        except:
            import traceback
            traceback.print_exc()

    @commands.slash_command(description="Plot driver position changes in the race.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def positionchanges(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                              round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
        """Line graph per driver showing position for each lap."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        # Load the data
        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True)
        sc_laps, vsc_laps = stats.find_sc_laps(session.laps)
        red_laps = stats.find_red_laps(session.laps)
        fig = Figure(figsize=(8.5, 5.46), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        # Plot the drivers position per lap
        for d in session.drivers:
            laps = await asyncio.to_thread(lambda: session.laps.pick_drivers(d))
            try:
                id = laps["Driver"].iloc[0]
                style = fastf1.plotting.get_driver_style(identifier=id,
                                                         style=[
                                                             'color', 'linestyle'],
                                                         session=session)
                ax.plot(laps["LapNumber"], laps["Position"], label=id,
                        **style)
            except:
                pass

        ax.set_title(
            f"Race Position - {ev['EventName']} ({ev['EventDate'].year})")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Position")
        ax.set_yticks(np.arange(1, len(session.drivers) + 1))
        ax.tick_params(axis="y", right=True, left=True,
                       labelleft=True, labelright=False)
        ax.invert_yaxis()
        stats.shade_sc_periods(sc_laps, vsc_laps, ax)
        stats.shade_red_flag(red_laps, ax)
        ax.legend(bbox_to_anchor=(1.01, 1.0))
        ax.grid(which="minor", alpha=0.1)
        ax.minorticks_on()

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
    async def fastestlaps(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                          round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds),
                          session: discord.Option(str, "Select the session",  autocomplete=resolve_sessions)):
        """Bar chart for each driver's fastest lap in `session`."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        session = stats.convert_shootout_to_qualifying(year, session)

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
        ax.grid(which="minor", alpha=0.1)
        ax.minorticks_on()

        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Fastest Laps: {ev.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="team-pace-delta", description="Rank team’s race pace from the fastest to the slowest.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def team_pace(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                        round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        race = await stats.load_session(ev, "R", laps=True, telemetry=True)
        laps = await asyncio.to_thread(lambda: race.laps.pick_quicklaps())

        transformed_laps = laps.copy()
        transformed_laps.loc[:,
                             "LapTime"] = laps["LapTime"].dt.total_seconds()

# order the team from the fastest (lowest median lap time) tp slower
        team_order = (
            transformed_laps[["Team", "LapTime"]]
            .groupby("Team")
            .median()["LapTime"]
            .sort_values()
            .index
        )

        def format_seconds(seconds):
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02}:{secs:06.3f}"

        # Format the y-axis ticks to display minutes and seconds
        def format_func(value, _):
            return format_seconds(value)
        team_palette = {team: fastf1.plotting.get_team_color(
            team, race) for team in team_order}
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(
            data=transformed_laps,
            x="Team",
            y="LapTime",
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
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

        ax.set(xlabel=None)
        plt.tight_layout()
        ax.grid(which="minor", alpha=0.1)
        ax.minorticks_on()

        file = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Team Pace delta: {ev.EventName} ',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="driver-lap-time-distribution", description="View driver(s) laptime distribution on track.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def driver_laps(self, ctx: ApplicationContext,
                          year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                          round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds),
                          driver: discord.Option(
                              str, "Select the driver", autocomplete=resolve_drivers)
                          ):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        if driver is None:
            raise ValueError("Specify a driver.")

        # Load laps and telemetry data
        ev = await stats.to_event(year, round)
        race = await stats.load_session(ev, "R", laps=True, telemetry=True)
        driver = utils.find_driver(driver, await ergast.get_all_drivers(year, ev["RoundNumber"]))["code"]
        try:
            driver_laps = await asyncio.to_thread(lambda: race.laps.pick_drivers(
                driver).pick_quicklaps().reset_index())
        except KeyError:
            await ctx.respond("No data for the driver found.\n-# *Why? The driver might not have participated in the given session.*")
            return
        driver_laps["LapTime"] = driver_laps["LapTime"].dt.total_seconds()

        # Handle compound mapping based on year
        if year == 2018:
            compound_label_mapping = {
                "SOFT": " ", "MEDIUM": " ", "HARD": " ",
                "HYPERSOFT": " ", "ULTRASOFT": " ",
                "SUPERSOFT": " ", "SUPERHARD": " "
            }
        else:
            # Get absolute compounds dynamically
            absolute_compounds = await stats.get_compound_async(year, ev.EventName)
            compound_numbers = [int(s[1:]) for s in absolute_compounds]
            absolute_number_mapping = {i: j for i, j in zip(
                compound_numbers, absolute_compounds)}
            soft_compound = absolute_number_mapping.get(max(compound_numbers))
            hard_compound = absolute_number_mapping.get(min(compound_numbers))
            remaining_compound = next(compound for compound, number in absolute_number_mapping.items()
                                      if number != soft_compound and number != hard_compound)
            compound_label_mapping = {
                "SOFT": soft_compound,
                "MEDIUM": absolute_number_mapping.get(remaining_compound),
                "HARD": hard_compound
            }

        def format_seconds(seconds):
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02}:{secs:06.3f}"

        # Format the y-axis ticks to display minutes and seconds
        def format_func(value, _):
            return format_seconds(value)

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

        # Invert y-axis to reflect faster times at the top
        ax.invert_yaxis()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.grid(which="minor", alpha=0.1)
        ax.minorticks_on()
        handles, labels = ax.get_legend_handles_labels()

        new_labels = [
            f"{label} ({compound_label_mapping.get(label, label)})"
            if label != " " and label != compound_label_mapping.get(label, label)
            else label
            for label in labels
        ]
        for i in range(len(new_labels)):
            if new_labels[i].endswith("( )"):
                # Remove the " ( )" from the string
                new_labels[i] = new_labels[i].rstrip(" ( )")
            elif re.search(r"(.*) \(\1\)$", new_labels[i]):
                # Truncate the " (x)" part where x is the same as the preceding text
                new_labels[i] = re.sub(r"(.*) \(\1\)$", r"\1", new_labels[i])

        # Set the updated legend with new labels
        ax.legend(handles, new_labels, title="Compound")

        plt.suptitle(
            f"{driver.upper()} Laptimes in the {year} {ev['EventName']}")

        # Turn on major grid lines
        plt.grid(color='w', which='major', axis='both')
        sns.despine(left=True, bottom=True)

        plt.tight_layout()
        file = utils.plot_to_file(fig, "plot")

        # Send the plot image as part of a message
        embed = discord.Embed(
            title=f'Driver lap time distribution: {driver}',
            color=get_top_role_color(ctx.author)
        )
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="track-speed", description="View driver speed on track.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def track_speed(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                          round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds),
                          driver: discord.Option(str, "Select the driver", autocomplete=resolve_drivers)):
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
        try:

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
            points = np.array([rotated_pos_x, rotated_pos_y]
                              ).T.reshape(-1, 1, 2)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            speed = car["Speed"]
            del lap, car

        except KeyError:
            await ctx.respond("No data for the driver found.\n-# *Why? The driver might not have participated in the given session.*")
            return
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
    async def track_sectors(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                            round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds),
                            session: discord.Option(str, "Select the session",  autocomplete=resolve_sessions),
                            driver1:  discord.Option(str, "Select driver 1", autocomplete=resolve_drivers),
                            driver2:  discord.Option(str, "Select driver 2", autocomplete=resolve_drivers),
                            lap: discord.Option(int, "Select the lap (default fastest)", default=None, autocomplete=resolve_laps)):
        """Plot a track map showing where a driver was faster based on minisectors."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        session = stats.convert_shootout_to_qualifying(year, session)
        event = await stats.to_event(year, round)
        s = await stats.load_session(event, session, laps=True, telemetry=True)
        if lap and int(lap) > s.laps["LapNumber"].unique().max():
            raise ValueError("Lap number out of range.")
        try:
            f = await sectors_func(year, round, session, driver1, driver2, lap, event, s)
        except KeyError:
            await ctx.respond("No data for the driver found.\n-# *Why? The driver might not have participated in the given session.*")
            return
        embed = discord.Embed(title=f'Fastest Sectors comparison: {event.EventName}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(description="Show the position gains/losses per driver in the race.", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def gains(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                    round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
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
    async def tyre_choice(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                          round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds),
                          session: discord.Option(
                              str, "Select the session",  autocomplete=resolve_sessions)
                          ):
        """Plot the distribution of tyre compound for all laps in the session."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)
        session = stats.convert_shootout_to_qualifying(year, session)

        # Get lap data and count occurance of each compound
        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, session, laps=True)
        stints = await asyncio.to_thread(lambda: s.laps)
        stints = stints[stints["Compound"] != "UNKNOWN"]
        t_count = stints["Compound"].value_counts()
        added_compounds = set()

        if year == 2018:
            compound_label_mapping = {"SOFT": " ", "MEDIUM": " ", "HARD": " ", "HYPERSOFT": " ",
                                      "ULTRASOFT": " ",
                                      "SUPERSOFT": " ", "SUPERHARD": " "}
        else:
            absolute_compounds = await stats.get_compound_async(year, ev.EventName)
            compound_numbers = [int(s[1:]) for s in absolute_compounds]
            absolute_number_mapping = {i: j for i, j in zip(
                compound_numbers, absolute_compounds)}
            soft_compound = absolute_number_mapping.get(max(compound_numbers))
            hard_compound = absolute_number_mapping.get(min(compound_numbers))
            remaining_compound = next(compound for compound, number in absolute_number_mapping.items()
                                      if number != soft_compound and number != hard_compound)
            compound_label_mapping = {
                "SOFT": soft_compound,
                "MEDIUM": absolute_number_mapping.get(remaining_compound),
                "HARD": hard_compound
            }

        # Calculate percentages and sort
        t_percent = t_count / len(stints) * 100
        sorted_count = t_count.sort_values(ascending=False)
        sorted_percent = t_percent.loc[sorted_count.index]
        legend_labels = [
            f"{compound} ({compound_label_mapping.get(compound, compound)})" for compound in sorted_count.index]

        for i in range(len(legend_labels)):
            if legend_labels[i].endswith("( )"):
                # Remove the " ( )" from the string
                legend_labels[i] = legend_labels[i].rstrip(" ( )")
            elif re.search(r"(.*) \(\1\)$", legend_labels[i]):
                # Truncate the " (x)" part where x is the same as the preceding text
                legend_labels[i] = re.sub(
                    r"(.*) \(\1\)$", r"\1", legend_labels[i])
        clrs = [fastf1.plotting.get_compound_color(
            i, s) for i in sorted_count.index]

        fig = Figure(figsize=(8, 6), dpi=DPI, layout="constrained")
        ax = fig.add_subplot(aspect="equal")

        wedges, texts, autotexts = ax.pie(
            sorted_percent, colors=clrs, autopct="%1.1f%%", textprops={"color": "black"})

        ax.legend(wedges, legend_labels, title="Compounds",
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
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
                           year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                           round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds),
                           driver1:  discord.Option(str, "Select driver 1", autocomplete=resolve_drivers),
                           driver2:  discord.Option(str, "Select driver 2", autocomplete=resolve_drivers)):
        """Plot the lap times between two drivers for all laps, excluding pitstops and slow laps."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R", laps=True, telemetry=True)
        # Get driver codes from the identifiers given
        try:
            drivers = [utils.find_driver(d, await ergast.get_all_drivers(year, ev["RoundNumber"]))["code"]
                       for d in (driver1, driver2)]

            # Group laps using only quicklaps to exclude pitstops and slow laps
            laps = await asyncio.to_thread(lambda: s.laps.pick_drivers(drivers).pick_quicklaps())
            times = laps.loc[:, ["Driver", "LapNumber",
                                 "LapTime"]].groupby("Driver")
            del laps
        except KeyError:
            await ctx.respond("No data for the driver found.\n-# *Why? The driver might not have participated in the given session.*")
            return

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
            title=f'Laptime Comparison between two drivers: {driver1} vs {driver2}', color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="lap-distribution",
                            description="Violin plot comparing distribution of laptimes on different tyres.", integration_types={
                                discord.IntegrationType.guild_install,
                                discord.IntegrationType.user_install,
                            })
    async def lap_distribution(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                               round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
        """Plot a swarmplot and violin plot showing laptime distributions and tyre compound
        for the top 10 point finishers."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R", laps=True)

        # Get the point finishers
        point_finishers = await asyncio.to_thread(lambda: s.drivers[:10])

        laps = await asyncio.to_thread(lambda: s.laps.pick_drivers(
            point_finishers).pick_quicklaps().set_index("Driver"))
        # Convert laptimes to seconds for seaborn compatibility
        laps["LapTime"] = laps["LapTime"].dt.total_seconds()
        labels = [s.get_driver(d)["Abbreviation"] for d in point_finishers]
        compounds = laps["Compound"].unique()

        # Handle compound mapping based on year
        if year == 2018:
            compound_label_mapping = {
                "SOFT": " ", "MEDIUM": " ", "HARD": " ",
                "HYPERSOFT": " ", "ULTRASOFT": " ",
                "SUPERSOFT": " ", "SUPERHARD": " "
            }
        else:
            absolute_compounds = await stats.get_compound_async(year, ev.EventName)
            compound_numbers = [int(s[1:]) for s in absolute_compounds]
            absolute_number_mapping = {i: j for i, j in zip(
                compound_numbers, absolute_compounds)}
            soft_compound = absolute_number_mapping.get(max(compound_numbers))
            hard_compound = absolute_number_mapping.get(min(compound_numbers))
            remaining_compound = next(compound for compound, number in absolute_number_mapping.items()
                                      if number != soft_compound and number != hard_compound)
            compound_label_mapping = {
                "SOFT": soft_compound,
                "MEDIUM": absolute_number_mapping.get(remaining_compound),
                "HARD": hard_compound
            }

        # Define legend labels combining generic and absolute compound names
        compound_legend_labels = {
            compound: f"{compound} ({compound_label_mapping.get(compound, compound)})" for compound in compounds}

        # Function to format seconds for the y-axis
        def format_seconds(seconds):
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02}:{secs:06.3f}"

        # Format the y-axis ticks to display minutes and seconds
        def format_func(value, _):
            return format_seconds(value)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.violinplot(
            data=laps,
            x=laps.index,
            y="LapTime",
            inner=None,
            scale="area",
            order=labels,
            palette=[utils.get_driver_or_team_color(d, s) for d in labels]
        )

        # Plot the swarm plot with compounds
        sns.swarmplot(
            data=laps,
            x="Driver",
            y="LapTime",
            order=labels,
            hue="Compound",
            palette=[fastf1.plotting.get_compound_color(
                c, s) for c in compounds],
            linewidth=0,
            size=5
        )

        # Format the y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.grid(which="minor", alpha=0.1)
        ax.minorticks_on()

        ax.set_xlabel("Driver (Point Finishers)")
        ax.set_title(
            f"Lap Distribution - {ev['EventName']} ({ev['EventDate'].year})")

        sns.despine(left=True, right=True)

        # Update the legend with generic and absolute compound names
        handles, _ = ax.get_legend_handles_labels()
        legend_labels = [compound_legend_labels[compound]
                         for compound in compounds]
        for i in range(len(legend_labels)):
            if legend_labels[i].endswith("( )"):
                # Remove the " ( )" from the string
                legend_labels[i] = legend_labels[i].rstrip(" ( )")
            elif re.search(r"(.*) \(\1\)$", legend_labels[i]):
                # Truncate the " (x)" part where x is the same as the preceding text
                legend_labels[i] = re.sub(
                    r"(.*) \(\1\)$", r"\1", legend_labels[i])
        ax.legend(handles, legend_labels, title="Tyre Compounds")

        plt.tight_layout()
        file = utils.plot_to_file(fig, "plot")

        # Send the plot image as part of a message
        embed = discord.Embed(
            title=f'Lap distribution on violin plot: {ev.EventName}',
            color=get_top_role_color(ctx.author)
        )
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=file, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="tyre-degradation",
                            description="Plot the performance of each tyre compound based on the age of the tyre.", integration_types={
                                discord.IntegrationType.guild_install,
                                discord.IntegrationType.user_install,
                            })
    async def tyreperf(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                       round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
        """Plot a line graph showing the performance of each tyre compound based on the age of the tyre."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R", laps=True)

        data = await asyncio.to_thread(lambda: stats.tyre_performance(s))
        compounds = data["Compound"].unique()

        # Handle compound mapping based on year
        if year == 2018:
            compound_label_mapping = {
                "SOFT": " ", "MEDIUM": " ", "HARD": " ",
                "HYPERSOFT": " ", "ULTRASOFT": " ",
                "SUPERSOFT": " ", "SUPERHARD": " "
            }
        else:
            absolute_compounds = await stats.get_compound_async(year, ev.EventName)
            compound_numbers = [int(s[1:]) for s in absolute_compounds]
            absolute_number_mapping = {i: j for i, j in zip(
                compound_numbers, absolute_compounds)}
            soft_compound = absolute_number_mapping.get(max(compound_numbers))
            hard_compound = absolute_number_mapping.get(min(compound_numbers))
            remaining_compound = next(compound for compound, number in absolute_number_mapping.items()
                                      if number != soft_compound and number != hard_compound)
            compound_label_mapping = {
                "SOFT": soft_compound,
                "MEDIUM": absolute_number_mapping.get(remaining_compound),
                "HARD": hard_compound
            }

        fig = Figure(figsize=(10, 5), dpi=DPI, layout="constrained")
        ax = fig.add_subplot()

        # Plot each compound with the mapped labels
        for cmp in compounds:
            mask = data["Compound"] == cmp
            tyre_life = data.loc[mask, "TyreLife"].values
            lap_times = data.loc[mask, "Seconds"].values
            if compound_label_mapping.get(cmp) == " " or compound_label_mapping.get(cmp) is None:
                label = cmp
            else:
                label = f"{cmp} ({compound_label_mapping.get(cmp)})"

            ax.plot(
                tyre_life,
                lap_times,
                label=label,
                color=fastf1.plotting.get_compound_color(cmp, s),
            )

        del data

        # Format the y-axis ticks to display minutes and seconds
        def format_seconds(seconds):
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02}:{secs:06.3f}"

        def format_func(value, _):
            return format_seconds(value)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

        ax.set_xlabel("Tyre Life")
        ax.set_ylabel("Lap Time")
        ax.set_title(
            f"Tyre Performance - {ev['EventDate'].year} {ev['EventName']}")
        ax.legend(title="Tyre Compounds")
        ax.grid(which="minor", alpha=0.1)
        ax.minorticks_on()
        ax.grid(True, alpha=0.1)

        f = utils.plot_to_file(fig, "plot")

        # Send the plot image as part of a message
        embed = discord.Embed(
            title=f'Tyre degradation: {ev.EventName}',
            color=get_top_role_color(ctx.author)
        )
        embed.set_image(url="attachment://plot.png")
        embed.description = r"-# Methodology: The data is filtered to include laps within 105% of the fastest lap and grouped by Compound and TyreLife to calculate the mean lap time for each group."
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @ commands.slash_command(name="avg-lap-delta",
                             description="Bar chart comparing average time per driver with overall race average as a delta.",
                             integration_types={discord.IntegrationType.guild_install, discord.IntegrationType.user_install})
    async def avg_lap_delta(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                            round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
        """Get the overall average lap time of the session and plot the delta for each driver."""
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        s = await stats.load_session(ev, "R", laps=True)
        yr, rd = ev["EventDate"].year, ev["EventName"]

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
        ax.set_title(
            f"{yr} {rd}\nDelta to Average ({utils.format_timedelta(session_avg)})").set_fontsize(16)

        # Save the plot to a file
        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'Average lap delta: {ev.EventName} ',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx))

    @commands.slash_command(name="track-elevation", description="View driver 3D visualisation of the circuit", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def trackelevation(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                             round: discord.Option(str, "Select the round (event)", autocomplete=resolve_rounds)):
        round = roundnumber(round, year)[0]
        year = roundnumber(round, year)[1]
        await utils.check_season(ctx, year)

        ev = await stats.to_event(year, round)
        session = await stats.load_session(ev, "R", laps=True, telemetry=True)
        lap = await asyncio.to_thread(lambda: session.laps.pick_fastest())

        z_values = lap.telemetry['Z']
        x = lap.telemetry['X'] / 182000
        y = lap.telemetry['Y'] / 182000
        z = z_values / 182000
        theta = np.radians(40)

        rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                    [0, 1, 0],
                                    [-np.sin(theta), 0, np.cos(theta)]])
        coords = np.vstack([x, y, z])
        rotated_coords = rotation_matrix.dot(coords)
        x_rot = rotated_coords[0, :]
        y_rot = rotated_coords[1, :]
        z_rot = rotated_coords[2, :]
        min_x, max_x = x_rot.min(), x_rot.max()
        min_y, max_y = y_rot.min(), y_rot.max()
        min_z, max_z = z_rot.min(), z_rot.max()
        valid_range = (x_rot > min_x) & (x_rot < max_x) & (y_rot > min_y) & \
            (y_rot < max_y) & (z_rot > min_z) & (z_rot < max_z)

        x_trimmed = x_rot[valid_range]
        y_trimmed = y_rot[valid_range]
        z_trimmed = z_rot[valid_range]
        thicc = (z_trimmed / 70) + 10
        segments = np.stack((np.c_[x_trimmed[:-1], x_trimmed[1:]],
                            np.c_[y_trimmed[:-1], y_trimmed[1:]],
                            np.c_[z_trimmed[:-1], z_trimmed[1:]]), axis=2)

        fig = plt.figure(figsize=(12, 10))
        ax_3d = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmax=z_trimmed.max(), vmin=z_trimmed.min()
                             )

        for i, segment in enumerate(segments):
            color = cmap(norm(z_trimmed[i]))

            ax_3d.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                       color=color, linewidth=thicc[i], alpha=0.3)

        ax_3d.plot(x_trimmed, y_trimmed, z_trimmed + 0.001,
                   color='black', linestyle='-', linewidth=3, zorder=5)
        ax_3d.set_xlabel("X Position (m)")
        ax_3d.set_ylabel("Y Position (m)")
        ax_3d.set_zlabel("Elevation (m)")

        ax_3d.set_xlim([x_trimmed.min(), x_trimmed.max()])
        ax_3d.set_ylim([y_trimmed.min(), y_trimmed.max()])
        ax_3d.set_zlim([z_trimmed.min(), z_trimmed.max()])
        ax_3d.grid(False)
        ax_3d.set_axis_off()

        # Title
        ax_3d.set_title(
            f"{ev['EventDate'].year} {ev['EventName']} - 3D Track Layout")
        ax_3d.view_init(elev=50, azim=200)
        cbar = fig.colorbar(plt.cm.ScalarMappable(
            norm=norm, cmap=cmap), ax=ax_3d)
        median_z = np.median(z_values)
        import builtins
        elev_label = f"{builtins.round(median_z/10)}"
        tick_labels = [f"{label*18200:.3f}" for label in cbar.get_ticks()]
        tick_labels.reverse()
        cbar.set_ticklabels(tick_labels)
        cbar.set_ticks([])
        cbar.ax.invert_yaxis()
        if elev_label.startswith("-"):
            cbar.set_label(
                f"<-- Decreasing | Elevation (relative to {elev_label[1:]} m) | Increasing -->")
        else:
            cbar.set_label(
                f"<-- Decreasing | Elevation (relative to {elev_label} m) | Increasing -->")
        z_values.dropna(inplace=True)
        fig.text(0.5, 0.01, f'Max elevation change: {(z_values.max()/10-z_values.min()/10):3f} m',
                 ha='center', va='center', fontsize=20)
        f = utils.plot_to_file(fig, "plot")
        embed = discord.Embed(title=f'3D Track Layout: {ev["EventName"]}',
                              color=get_top_role_color(ctx.author))
        embed.set_image(url="attachment://plot.png")
        embed.description = f"-# Current viewing angle in degrees: 50\n-# Figure scaled down by a factor of 182000 for better viewing experience."

        class MyModal(discord.ui.Modal):

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.add_item(discord.ui.InputText(
                    label="Enter angle in degs (Recommended: 30 to 90):"))

            async def callback(self, interaction: discord.Interaction):
                try:
                    angle = int(self.children[0].value)
                    await interaction.response.defer(ephemeral=get_ephemeral_setting(ctx))
                    fig = plt.figure(figsize=(12, 10))
                    ax_3d = fig.add_subplot(111, projection='3d')
                    cmap = plt.get_cmap('viridis')
                    norm = plt.Normalize(vmin=z_trimmed.min(),
                                         vmax=z_trimmed.max())

                    for i, segment in enumerate(segments):
                        color = cmap(norm(z_trimmed[i]))

                        ax_3d.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                                   color=color, linewidth=thicc[i], alpha=0.3)

                    ax_3d.plot(x_trimmed, y_trimmed, z_trimmed + 0.001,
                               color='black', linestyle='-', linewidth=3, zorder=5)
                    ax_3d.set_xlabel("X Position (m)")
                    ax_3d.set_ylabel("Y Position (m)")
                    ax_3d.set_zlabel("Elevation (m)")

                    ax_3d.set_xlim([x_trimmed.min(), x_trimmed.max()])
                    ax_3d.set_ylim([y_trimmed.min(), y_trimmed.max()])
                    ax_3d.set_zlim([z_trimmed.min(), z_trimmed.max()])
                    ax_3d.grid(False)
                    ax_3d.set_axis_off()

                    # Title
                    ax_3d.set_title(
                        f"{ev['EventDate'].year} {ev['EventName']} - 3D Track Layout")
                    ax_3d.view_init(elev=angle, azim=200)
                    cbar = fig.colorbar(plt.cm.ScalarMappable(
                        norm=norm, cmap=cmap), ax=ax_3d)
                    median_z = np.median(z_values)
                    import builtins
                    elev_label = f"{builtins.round(median_z/10)}"

                    tick_labels = [
                        f"{label:.3f}" for label in cbar.get_ticks()]
                    tick_labels.reverse()
                    cbar.set_ticklabels(tick_labels)
                    cbar.ax.invert_yaxis()
                    cbar.set_ticklabels(tick_labels)
                    cbar.set_ticks([])
                    if elev_label.startswith("-"):
                        cbar.set_label(
                            f"<-- Decreasing | Elevation (relative to {elev_label[1:]} m) | Increasing -->")
                    else:
                        cbar.set_label(
                            f"<-- Decreasing | Elevation (relative to {elev_label} m) | Increasing -->")
                    fig.text(0.5, 0.01, f'Max elevation change: {(z_values.max()/10-z_values.min()/10):3f} m',
                             ha='center', va='center', fontsize=20)
                    f = utils.plot_to_file(fig, "plot")
                    embed = discord.Embed(title=f'3D Track Layout: {ev["EventName"]}',
                                          color=get_top_role_color(ctx.author))
                    embed.set_image(url="attachment://plot.png")
                    embed.description = f"-# Current viewing angle in degrees: {angle}\n-# Figure scaled down by a factor of 182000 for better viewing experience."
                    await interaction.edit(file=f, embed=embed)
                except ValueError:
                    await interaction.respond("Enter a valid angle (in degrees)", ephemeral=True)

        class MyView(discord.ui.View):
            @discord.ui.button(label="Change viewing angle", style=discord.ButtonStyle.primary)
            async def button_callback(self, button, interaction):
                await interaction.response.send_modal(MyModal(title="Change viewing angle"))
        await ctx.respond(embed=embed, file=f, ephemeral=get_ephemeral_setting(ctx), view=MyView(timeout=None))

    @commands.slash_command(name="quali-gap", description="View average or median quali gaps between teammates over a season", integration_types={
        discord.IntegrationType.guild_install,
        discord.IntegrationType.user_install,
    })
    async def qualigap(self, ctx: ApplicationContext, year: discord.Option(int, "Select the season", autocomplete=resolve_years_fastf1),
                       session: discord.Option(str, "Select the sssion", autocomplete=options.resolve_sessions_by_year_quali), type: options.AvgMedianOption):
        await utils.check_season(ctx, year)
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        sess = session

        if session == "Sprint Shootout" and year == 2023:
            schedule = schedule[schedule['EventFormat'] == 'sprint_shootout']
            scheduleiteration = schedule['RoundNumber'].tolist()
            result_setting = 'sprint_shootout'
        elif session == "Sprint Qualifying" and year >= 2024:
            schedule = schedule[schedule['EventFormat'] == 'sprint_qualifying']
            scheduleiteration = schedule['RoundNumber'].tolist()
            result_setting = 'sprint_qualifying'
        else:
            scheduleiteration = schedule['RoundNumber'].tolist()
            result_setting = 'Qualifying'
        tasks = []

        async def plot_team_gaps(team_gaps, session_results, calculation_type, year):
            session = fastf1.get_session(year, 1, "R")

            # Prepare data
            x_labels = []
            gap_values = []
            colors = []

            for team_name, gaps in team_gaps.items():
                if not gaps:
                    continue

                # Get the drivers for this team
                drivers = session_results[session_results["TeamName"] == team_name].sort_values(
                    "ClassifiedPosition")
                if len(drivers) < 2:
                    continue  # Skip teams with less than two drivers

                driver_1, driver_2 = drivers.iloc[0], drivers.iloc[1]

                if calculation_type == 'Average':
                    gap_value = np.mean(gaps)
                else:  # Median
                    gap_value = np.median(gaps)

                # Determine who has the better performance
                better_driver = driver_2 if gaps[0] < 0 else driver_1
                better_driver_code = better_driver["Abbreviation"]

                # Annotate x-axis labels
                x_labels.append(
                    f"{team_name} (in favor of {better_driver_code})")
                gap_values.append(gap_value)

                # Get team color
                team_color = fastf1.plotting.get_team_color(team_name, session)
                colors.append(team_color)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(x_labels, gap_values, color=colors)

            # Annotate bars with gap values
            for bar, gap in zip(bars, gap_values):
                ax.text(
                    bar.get_width() + 0.01,  # Position to the right of the bar
                    bar.get_y() + bar.get_height() / 2,  # Centered vertically
                    f"{gap:.3f}s",  # Rounded to 3 decimal places
                    va='center',
                    fontsize=10,
                    color='white'
                )

            ax.set_xlabel("Qualifying Gap (seconds)")
            ax.set_ylabel("Teams")
            ax.grid(which="minor", alpha=0.1)
            ax.minorticks_on()
            plt.tight_layout()

            return fig, ax
        for c in scheduleiteration:

            event = await stats.to_event(year, c)
            if result_setting in ['sprint_qualifying', 'sprint_shootout']:

                task = stats.load_session(event, session, laps=True, telemetry=False,
                                          weather=False, messages=True)
            else:
                task = stats.load_session(event, session, laps=False, telemetry=False,
                                          weather=False, messages=False)
            tasks.append(task)

        team_gaps = {}
        results = await asyncio.gather(*tasks)

        for session in results:
            session_results = session.results  # The DataFrame subclass

            # Group by team
            teams = session_results.groupby("TeamName")

            for team_name, drivers in teams:
                if len(drivers) < 2:
                    continue  # Skip if less than two drivers in the team

                # Sort drivers by their classified position
                drivers = drivers.sort_values("ClassifiedPosition")

                driver_1 = drivers.iloc[0]
                driver_2 = drivers.iloc[1]

                # Store qualifying gaps
                gaps = []

                for quali_phase in ["Q3", "Q2", "Q1"]:
                    time_1 = driver_1[quali_phase]
                    time_2 = driver_2[quali_phase]

                    if pd.notna(time_1) and pd.notna(time_2):
                        gap = abs(time_1.total_seconds() -
                                  time_2.total_seconds())
                        gaps.append(gap)
                        break  # Only compare the same phase

                if team_name not in team_gaps:
                    team_gaps[team_name] = []

                team_gaps[team_name].extend(gaps)

        if type == 'Average':
            fig, ax = await plot_team_gaps(team_gaps, session_results, calculation_type='Average', year=year)
            ax.set_title(
                f"Teammate {type} {sess} Gaps")
            f = utils.plot_to_file(fig, "plot")
            embed = discord.Embed(
                title=f'Teammate Average {sess} gaps: {year}', color=get_top_role_color(ctx.author))
            embed.set_image(url="attachment://plot.png")
            embed.description = "-# Methodology: The teammate gaps are calculated based on the times set by the drivers in the same session (eg. Q1, Q2, Q3). Wet sessions are **NOT** excluded. "
            await ctx.respond(file=f, embed=embed, ephemeral=get_ephemeral_setting(ctx))

        else:
            fig, ax = await plot_team_gaps(team_gaps, session_results, calculation_type='Median', year=year)
            ax.set_title(
                f"Teammate {type} {sess} Gaps")
            f = utils.plot_to_file(fig, "plot")
            embed = discord.Embed(
                title=f'Teammate Median {sess} gaps: {year}', color=get_top_role_color(ctx.author))
            embed.set_image(url="attachment://plot.png")
            embed.description = "-# Methodology: The teammate gaps are calculated based on the times set by the drivers in the same session (eg. Q1, Q2, Q3). Wet sessions are **NOT** excluded. "
            await ctx.respond(file=f, embed=embed, ephemeral=get_ephemeral_setting(ctx))


def setup(bot: discord.Bot):
    bot.add_cog(Plot(bot))
