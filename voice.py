# from playsound import playsound
import os  
def playvoice(file):
     os.system("vlc --play-and-exit ./voice/"+file)