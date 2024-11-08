import discord

import random
class Embed:
    
    # embed.set_image(url=None)
    # embed.set_footer(text='')
    # embed.description = ''
    def __init__(self, title = None, description = None, colour = None, image_url = None,thumbnail_url = None,author = None, footer = None):
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
        if  (not (author == None) and len(author) == 2):
            self.embed.set_author(name=author[0], icon_url=author[1])
        if not (colour == None):
            self.embed.colour = colour
        if not (image_url == None):
            self.embed.set_image(url = image_url)
        if not (thumbnail_url == None):
            self.embed.set_thumbnail(url=thumbnail_url)
        if not (footer == None):
            self.embed.set_footer(text=footer)
class ErrorEmbed(Embed):
    def __init__(self,title = None, error_message = None,footer_message = None):
        self.embed = discord.Embed(title=f"Error Occured :(", description="Default Error Message")
        
        
        self.embed.clear_fields()
        
        
        super().__init__(title=title,
                         description=str(error_message),   
                         footer=footer_message
                         )
class OffseasonEmbed(Embed):
    def __init__(self):
        self.embed = discord.Embed(title=f"Offseason Embed", description="",).set_thumbnail(url='https://cdn.discordapp.com/attachments/884602392249770087/1059464532239581204/f1python128.png')
        self.embed.set_author(name='f1buddy',icon_url='https://raw.githubusercontent.com/F1-Buddy/f1buddy-python/main/botPics/f1pythonpfp.png')

        self.embed.clear_fields()
        gif_urls = ['https://media.tenor.com/kdIoxRG4W4QAAAAC/crying-crying-kid.gif',
                    'https://media1.tenor.com/m/Aopm1M7LJSUAAAAd/toto-smash-toto-wolff.gif',
                    'https://media1.tenor.com/m/SM_22nlNWhkAAAAC/fernando-alonso-alonso.gif',
                    'https://media1.tenor.com/m/9atYe_gKtbMAAAAd/sainz-sainz-jr.gif',
                    'https://media1.tenor.com/m/YlbGgdszkpEAAAAC/george-russel-f1.gif',
                    ]
        super().__init__(title='Race Schedule', 
                         description='It is currently off season! :crying_cat_face:', 
                         image_url=random.choice(gif_urls),
                         footer="Schedule will be available 2 weeks before the first GP"
                         )