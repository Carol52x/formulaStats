![image](https://cdn.discordapp.com/avatars/1290361750520070225/b4da700bcdd5577d9a80d16bf40a98ab.png)

# formulaStats

formulaStats is a discord bot implementation to view Formula 1 statistics and other visuals inside discord embeds via slash commands. formulaStats sources its data from [FastF1](https://github.com/theOehrly/Fast-F1) and [Jolpica (Now deprecated Ergast's successor)](https://github.com/jolpica/jolpica-f1), and uses [Pycord](https://github.com/Pycord-Development/pycord) to interact with the discord API.


## Usage

Invite the bot via [this link](https://discord.com/oauth2/authorize?client_id=1290361750520070225). 

You have the option to either add it your account (which allows you to use the bot anywhere, including DMs) or to a server. Please note that quiz functionality and toggling silent mode (aka ephemeral messaging) is only available when you add it to a server and you **require** adminstrator privileges in the server to set up these features, for safety measures. 

When you add it to a server, silent-mode is enabled by default. Use `/help` to find the command list and documentation. When you use it via user-install, the messages are by default ephemeral to avoid `discord.Forbidden` exceptions.

While using the slash commands, by default, the  `year` parameter is the current season, the `round` parameter is the last *completed* race weekend and the `session` parameter is the main race. Other commands have extra parameters where the default parameters are documentioned in the description area.  `round` parameter can either be the round number of the event, the circuit name or the name of the country where the event takes place. `Driver` parameter(s) can either be the last name, driver number or the three-letter code of the driver. However, please note `/career` **requires** full name of the driver to avoid scenarios where two different drivers share the same last name.


## Some Notes

Some data (particularly which are sourced from Jolpica API) take a day or two after the race to update, while data sourced from fastf1 usually updates in about an hour after the given session ends. It is recommended to use `/generate-cache` for a particular F1 session to accelerate the plotting commands for that session as some of them take a while to complete. Some commands have a limited set of parameters to choose from due to the limitations of the aforementioned data sources. There are also some inherent inaccuracies (which probably do not have a quick fix solution at the moment) with the way some particular data is calculated and therefore, will be mentioned in the embed footer, if any. 

`/quiz` can only work after issuing `/quizsetup` and following the prompts, by an admin in the server. Same goes for `/silent-mode` as mentioned previously. Please note `/quiz` requires media permissions in the channel it is supposed to send a quiz in.

Potential bugs can be reported via opening an issue on this repository or DMing `carol520` on discord.



## Running your own instance

If you wish to run your own instance of the bot, here is an example setup of running the bot locally with python's virtual environment:

**Setting things up**

The application requires **[Python 3.11+](https://www.python.org/downloads/release/python-3110/)**, **[FFmpeg](https://www.ffmpeg.org/) (for the `/radio` command which converts audio files to video files for mobile discord users)** and **[Reddit developer credentials](https://developers.reddit.com/) (for the `/reddit`command which uses [AsyncPRAW](https://github.com/praw-dev/asyncpraw))**.

```bash
git clone https://github.com/Carol52x/formulaStats.git
cd formulaStats/
````

Rename `.env.example` to `.env`

Fill in `.env` file and optionally configure in `config.ini`

```bash
python -m venv f1bot 
f1bot\Scripts\activate
pip install -r requirements.txt
python -m main
```

**After install**

A `/cache` directory will be created in the project root when the bot is running. This may become large over time with session telemetry (~100 MB per race weekend). You can manually delete the `/cache` folder or specific subfolders, or a script is provided in the root directory: `python -m flushcache`. Make sure the bot is not running. A new cache will be created during the next startup. Some temp media files can also be generated if `/radio` is used and logs are stored in a `.txt` file in the logs folder. `bot_settings.db` and `guild_roles.db` are generated when `/silent-mode` and `/quizsetup` are used, respectively.

This bot comes with [CogWatch](https://github.com/robertwayne/cogwatch) out of the box to make development easier for the modules inside `f1/cogs` folder.





