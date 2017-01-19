# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:27:28 2015
this module is used for visualization
it's a package of pil module
and all the method is based on pil module
@author: teddy
"""
from PIL import ImageDraw,ImageFont

def fromIDgetcolor(Id):
    """
    you can input a int index number or a color name
    """
    colors ={
    'white':(0,0,0),
    'red':(255,0,0),
    'green':(0,255,0),
    'blue':(0,0,255),
    'cyan':(0,255,255),
    'magenta':(255,0,255),
    'yellow ':(212,212,0),
    'black ':(25,25,25),
    'forestgreen':(34,139,34),
    'deepskyblue':(0,191,255),
    'darkred':(139,0,0),
    'orchid':(218,112,214),
    'sandybrown':(244,164,96)
    }
    if isinstance(Id,int):
        idx = Id%len(colors)
        return list(colors.values())[idx]
    elif isinstance(Id,str):
        if colors.get(Id,None) is not None:
            return colors[Id]
        else:
            return colors['white']
    else:
        return colors['white']

def drawbox(im,xy,c_index = 'red',width=5):
    """
    you can set the color and the width of bbox
    the order of coordinate should be leftx,rightx,upy,downy
    """
    draw = ImageDraw.Draw(im)
    color = fromIDgetcolor(c_index)
    u_l = [xy[0],xy[2],xy[1],xy[2]]
    d_l = [xy[0],xy[3],xy[1],xy[3]]
    l_l = [xy[0],xy[2],xy[0],xy[3]]
    r_l = [xy[1],xy[2],xy[1],xy[3]]
    draw.line(u_l,color,width)
    draw.line(d_l,color,width)
    draw.line(l_l,color,width)
    draw.line(r_l,color,width)
    del draw
    
def drawtext(im,xy,my_str,c_index = 'red',font = 'ukai.ttc',size = 60):
    """
    coodinate is the left_up point of the string 
    """
    draw = ImageDraw.Draw(im)
    color = fromIDgetcolor(c_index)
    font = ImageFont.truetype(font, size)
    draw.text([xy[0],xy[1]],my_str,color,font = font)
    del draw

def drawpoint(im,xy,c_index = 'green',radius = 5):
    """
    the size is point's radius,xy is the center of of the circle,first is x,second is y
    """
    draw = ImageDraw.Draw(im)
    color  = fromIDgetcolor(c_index)
    draw.pieslice([xy[0]-radius,xy[1]-radius,xy[0]+radius,xy[1]+radius],0,360,
                  color) #first give a bouding box of a circle
    del draw