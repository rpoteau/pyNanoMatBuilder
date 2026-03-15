##################################################################################################
#
#                              p y N a n o M a t B u i l d e r
#
#    I N - H O U S E  I N I T I A L I Z I N G     F U N C T I O N S    a n d    C L A S S E S
#
##################################################################################################

__author__ = "Romuald Poteau"
__maintainer__ =  "Romuald Poteau"
__email__ = "romuald.poteau@utoulouse.fr"
__status__ = "Development"

import os,sys,platform
from datetime import datetime, timedelta
import datetime, time
#from IPython.core.display import display,Image,Markdown,HTML
from IPython.display import display,Image,Markdown,HTML
from urllib.request import urlopen
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
DEFAULT_RES_PATH = PACKAGE_ROOT / "resources"

_start_time   = None
_end_time     = None
_chrono_start = None
_chrono_stop  = None


def apply_css_style():
    """
    Explicitly reads and applies the visualID CSS stylesheet 
    from the package resources.
    """
    path2css = DEFAULT_RES_PATH / "css" / "visualID.css"
    if path2css.exists():
        with open(path2css, "r") as f:
            styles = f.read()
        # Some CSS files might not have the <style> tag if they are raw CSS
        if not styles.strip().startswith("<style>"):
            styles = f"<style>{styles}</style>"
        display(HTML(styles))
    else:
        print(f"[Warning] CSS file not found at {path2css}")

def display_md(text):
    display(Markdown(text))
    
def hdelay(sec):
    return str(datetime.timedelta(seconds=int(sec)))    
    
# Return human delay like 01:14:28 543ms
# delay can be timedelta or seconds
def hdelay_ms(delay):
    if type(delay) is not timedelta:
        delay=timedelta(seconds=delay)
    sec = delay.total_seconds()
    hh = sec // 3600
    mm = (sec // 60) - (hh * 60)
    ss = sec - hh*3600 - mm*60
    ms = (sec - int(sec))*1000
    return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'

def init():
    """
    Initializes the notebook environment: applies CSS, 
    displays the banner, and shows hostname/time.
    """
    global _start_time
    _start_time = datetime.datetime.now()

    # 1. Call the explicit CSS function
    apply_css_style()
    
    # 2. Display the banner
    path2banner = DEFAULT_RES_PATH / 'svg' / 'pyNanoMatBuilder_banner.svg'
    
    if path2banner.exists():
        with open(path2banner, "r") as f: svg_data = f.read()
        display(HTML(f'<div style="text-align: center; max-width: 1200px; margin: 0 auto; height: auto;">{svg_data}</div>'))
    else:
        print(f"[Warning] banner file not found at {path2banner}")
    
    # 3. Environment Info
    now = datetime.datetime.now().strftime("%A %d %B %Y, %H:%M:%S")
    display(Markdown(f"**Environment initialized:** {now} on {platform.node()}"))

def end():
    """
    Terminates the notebook session: displays duration, 
    end time, and the termination logo from package resources.
    """
    global _start_time, _end_time
    
    # 1. Calcul du temps
    _end_time = datetime.datetime.now()
    end_str = _end_time.strftime("%A %d %B %Y, %H:%M:%S")
    
    # Calcul de la durée si _start_time existe
    if _start_time:
        duration = hdelay_ms(_end_time - _start_time)
    else:
        duration = "Unknown (init() was not called)"

    # 2. Affichage des infos de fin
    md = f'**End at:** {end_str}  \n'
    md += f'**Duration:** {duration}'

    # 3. Affichage du logo de fin (depuis le package)
    # On suppose que le logo s'appelle logoEnd.svg et est dans resources/svg/
    path2logo = DEFAULT_RES_PATH / "svg" / "logoEnd.svg"
    
    if path2logo.exists():
        with open(path2logo, "r") as f:
            svg_data = f.read()
        display(HTML(f'<div style="text-align: center; width: 100%; margin: auto;">{svg_data}</div>'))
    else:
        # Fallback si le logo est manquant
        print(f"[Warning] End logo not found at {path2logo}")
    display_md(md)
    

