###############################################################################
#
#                      p y N a n o M a t B u i l d e r
#
#       I N - H O U S E   F U N C T I O N S    a n d    C L A S S E S
###############################################################################

__author__ = "Romuald POTEAU"
__maintainer__ =  "Romuald POTEAU"
__email__ = "romuald.poteau@univ-tlse3.fr"
__status__ = "Development"

def centerTitle(content=None):
    '''
    centers and renders as HTML a text in the notebook
    font size = 16px, background color = dark grey, foreground color = white
    '''
    from IPython.display import display, HTML
    display(HTML(f"<div style='text-align:center; font-weight: bold; font-size:16px;background-color: #343132;color: #ffffff'>{content}</div>"))

def centertxt(content=None,font='sans', size=12,weight="normal",bgc="#000000",fgc="#ffffff"):
    '''
    centers and renders as HTML a text in the notebook
    input: 
        - content = the text to render (default: None)
        - font = font family (default: 'sans', values allowed =  'sans-serif' | 'serif' | 'monospace' | 'cursive' | 'fantasy' | ...)
        - size = font size (default: 12)
        - weight = font weight (default: 'normal', values allowed = 'normal' | 'bold' | 'bolder' | 'lighter' | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 )
        - bgc = background color (name or hex code, default = '#ffffff')
        - fgc = foreground color (name or hex code, default = '#000000')
    '''
    from IPython.display import display, HTML
    display(HTML(f"<div style='text-align:center; font-family: {font}; font-weight: {weight}; font-size:{size}px;background-color: {bgc};color: {fgc}'>{content}</div>"))

import os,sys,platform
from datetime import datetime
import datetime, time
#from IPython.core.display import display,Image,Markdown,HTML
from IPython.display import display,Image,Markdown,HTML
from urllib.request import urlopen

_start_time   = None
_end_time     = None
_chrono_start = None
_chrono_stop  = None

class fg:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    LIGHTGRAY = "\033[37m"
    DARKGRAY = "\033[90m"    
    BLACK = '\033[30m'
    WHITE = "\033[38;5;231m"
    OFF = '\033[0m'
class hl:
    BLINK = "\033[5m"
    blink = "\033[25m" #reset blink
    BOLD = '\033[1m'
    bold = "\033[21m" #reset bold
    UNDERL = '\033[4m'
    underl = "\033[24m" #reset underline
    ITALIC = "\033[3m"
    italic = "\033[23m"
    OFF = '\033[0m'
class bg:
    DARKRED = "\033[38;5;231;48;5;52m"
    DARKREDB = "\033[38;5;231;48;5;52;1m"
    LIGHTRED = "\033[48;5;217m"
    LIGHTREDB = "\033[48;5;217;1m"
    LIGHTYELLOW = "\033[48;5;228m"
    LIGHTYELLOWB = "\033[48;5;228;1m"
    LIGHTGREEN = "\033[48;5;156m"
    LIGHTGREENB = "\033[48;5;156;1m"
    LIGHTBLUE = "\033[48;5;117m"
    LIGHTBLUEB = "\033[48;5;117;1m"
    OFF = "\033[0m"
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    LIGHTGRAY = "\033[37m"
    DARKGRAY = "\033[90m"    
    BLACK = '\033[30m'
    WHITE = "\033[38;5;231m"
    BOLD = '\033[1m'
    OFF = '\033[0m'

def css_styling(pwy):
    styles = open(pwy+"/css/visualID.css", "r").read()
    display(HTML(styles))
#def css_styling():
#    html = urlopen("file:./css/visualID.css")
#    styles = HTML(html.read().decode('utf-8'))
#    display(HTML(styles))

def display_md(text):
    display(Markdown(text))
    
def hdelay(sec):
    return str(datetime.timedelta(seconds=int(sec)))    
    
# Return human delay like 01:14:28 543ms
# delay can be timedelta or seconds
def hdelay_ms(delay):
    if type(delay) is not datetime.timedelta:
        delay=datetime.timedelta(seconds=delay)
    sec = delay.total_seconds()
    hh = sec // 3600
    mm = (sec // 60) - (hh * 60)
    ss = sec - hh*3600 - mm*60
    ms = (sec - int(sec))*1000
    return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'

def chrono_start():
    global _chrono_start, _chrono_stop
    _chrono_start=time.time()

# return delay in seconds or in human format
def chrono_stop(hdelay=False):
    global _chrono_start, _chrono_stop
    _chrono_stop = time.time()
    sec = _chrono_stop - _chrono_start
    if hdelay : return hdelay_ms(sec)
    return sec

def chrono_show():
    print('\nDuration : ', hdelay_ms(time.time() - _chrono_start))

def init(pwy):
    global _start_time
    # Styling notebook
    #
    css_styling(pwy)
    # Today, now and hostname
    #
    _start_time = datetime.datetime.now()
    start_time = _start_time.strftime("%A %d %B %Y, %H:%M:%S")
    _h = platform.uname()
    h = _h[1]+" ("+_h[0]+")"
    md = f'**Start at:** {start_time}  \n'
    md+= f'**Hostname:** {h}'
    display_md(md)
    #print('Run time             :', _start_time.strftime("%A %d %B %Y, %H:%M:%S"))
    #print('Hostname             :', f'{h[1]} ({h[0]})')
    path2svg=pwy + 'svg/'
    md = '<p style="text-align: center"><img width="800px" src="' + path2svg + 'pyNanoMatBuilder_banner.svg" style="margin-left:auto; margin-right:auto"/></p>'
    display_md(md)
    
def end(pwy):
    global _end_time
    _end_time = datetime.datetime.now()
    end_time = time.strftime("%A %d %B %Y, %H:%M:%S")
    duration = hdelay_ms(_end_time - _start_time)
    md = f'**End at:** {end_time}  \n'
    md+= f'**Duration:** {duration}'
    display_md(md)
    path2svg=pwy + 'svg/'
    md = '<p style="text-align: center"><img width="800px" src="' + path2svg + 'logoEnd.svg" style="margin-left:auto; margin-right:auto"/></p>'
    display_md(md)

