#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:02:24 2023

Class related to making figure-quality plots

Probably best to set your default sans-serif font to Helvetica before you make
figures: https://fowlerlab.org/2019/01/03/changing-the-sans-serif-font-to-helvetica/
 
The maximum published width for a one-column
figure is 3.33 inches (240 pt). The maximum width for a two-column
figure is 6.69 inches (17 cm). The maximum depth of figures should 
be 8 ¼ in. (21.1 cm).

panel labels are 8 point font, ticks are 7 point font,
annotations and legends are 6 point font

@author: jbeckwith
"""
import matplotlib # requires 3.8.0
import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def __init__(self, poster=False):
        self.poster = poster
        self = self
        return
    
    def two_column_plot(self, nrows=1, ncolumns=1, heightratio=[1], widthratio=[1], height=0):
        """ two_column_plot function
        takes data and makes a two-column width figure
        ================INPUTS============= 
        npanels is panel matrix
        ratios is size ratios of panels
        ================OUTPUT============= 
        fig, axs are figure objects """
        
        # first, check everything matches
        try:
            if len(heightratio) != nrows:
                raise Exception('Number of height ratios incorrect')
            if len(widthratio) != ncolumns:
                raise Exception('Number of width ratios incorrect')
        except Exception as error:
            print('Caught this error: ' + repr(error))
            return
        
        if self.poster == True:
            fontsz = 12
            lw = 1
        else:
            fontsz = 7
            lw = 1

        xsize = 6.69 # 3.33 inches for one-column figure
        if height == 0:
            ysize = np.min([3.5*nrows, 8.25]) # maximum size in y can be 8.25
        else:
            ysize = height
        
        plt.rcParams['figure.figsize'] = [xsize, ysize]
        plt.rcParams['font.size'] = fontsz
        plt.rcParams['svg.fonttype'] = 'none'
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.rcParams['axes.linewidth'] = lw # set the value globally
        
        fig, axs = plt.subplots(nrows, ncolumns, height_ratios=heightratio, 
                                width_ratios=widthratio) # create number of panels


        # clean up axes, tick parameters
        if nrows*ncolumns == 1:
            axs.xaxis.set_tick_params(width=lw, length=lw*4)
            axs.yaxis.set_tick_params(width=lw, length=lw*4)
            axs.tick_params(axis='both', pad=1.2)
        elif nrows*ncolumns == 2:
            for i in np.arange(2):
                 axs[i].xaxis.set_tick_params(width=lw, length=lw*4)
                 axs[i].yaxis.set_tick_params(width=lw, length=lw*4)
                 axs[i].tick_params(axis='both', pad=1.2)
        elif nrows*ncolumns == len(widthratio):
            for i in np.arange(len(widthratio)):
                 axs[i].xaxis.set_tick_params(width=lw, length=lw*4)
                 axs[i].yaxis.set_tick_params(width=lw, length=lw*4)
                 axs[i].tick_params(axis='both', pad=1.2)            
        else:
            for i in np.arange(nrows):
                for j in np.arange(ncolumns):
                    axs[i,j].xaxis.set_tick_params(width=0.5, length=lw*4)
                    axs[i,j].yaxis.set_tick_params(width=0.5, length=lw*4)
                    axs[i,j].tick_params(axis='both', pad=1.2)
        return fig, axs

    def one_column_plot(self, npanels=1, ratios=[1], height=None):
        """ one_column_plot function
        takes data and makes a one-column width figure
        ================INPUTS============= 
        npanels is number of panels in the figure
        ratios is size ratios of panels
        height is figure height
        ================OUTPUT============= 
        fig, axs are figure objects """
        
        # first, check everything matches
        try:
            if len(ratios) != npanels:
                raise Exception('Number of ratios incorrect')
        except Exception as error:
            print('Caught this error: ' + repr(error))
            return

        if self.poster == True:
            fontsz = 12
            lw = 1
        else:
            fontsz = 7
            lw = 0.5

        xsize = 3.33 # 3.33 inches for one-column figure
        if height is not None:
            ysize = np.min([height, 8.25]) # maximum size in y can be 8.25
        else:
            ysize = np.min([3.5*npanels, 8.25]) # maximum size in y can be 8.25
            
        plt.rcParams['figure.figsize'] = [xsize, ysize]
        plt.rcParams['font.size'] = fontsz
        plt.rcParams['svg.fonttype'] = 'none'
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.rcParams['axes.linewidth'] = lw # set the value globally
        
        fig, axs = plt.subplots(npanels, 1, height_ratios=ratios) # create number of panels


        # clean up axes, tick parameters
        if npanels == 1:
            axs.xaxis.set_tick_params(width=lw, length=lw*4)
            axs.yaxis.set_tick_params(width=lw, length=lw*4)
            axs.tick_params(axis='both', pad=1.2)
        else:
            for i in np.arange(npanels):
                axs[i].xaxis.set_tick_params(width=lw, length=lw*4)
                axs[i].yaxis.set_tick_params(width=lw, length=lw*4)
                axs[i].tick_params(axis='both', pad=1.2)      
        return fig, axs
    
    def line_plot(self, axs, x, y, xlim=None, ylim=None,
        color='k', lw=0.75, label='', xaxislabel='x axis', yaxislabel='y axis', ls='-'):
        """ line_plot function
        takes data and makes a line plot
        ================INPUTS============= 
        x is x data
        y os y data
        xlim is x limits; default is None (which computes max/min)
        ylim is y limits; default is None (which computes max/min)
        color is line colour; default is black
        lw is line width (default 0.75)
        label is label; default is nothing
        xaxislabel is x axis label (default is 'x axis')
        yaxislabel is y axis label (default is 'y axis')
        ================OUTPUT============= 
        axs is axis object """
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8
        
        if xlim is None:
            xlim = np.array([np.min(x), np.max(x)])
        if ylim is None:
            ylim = np.array([np.min(y), np.max(y)])
        axs.plot(x, y, lw=lw, color=color, label=label, ls=ls)
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        axs.grid(True,which="both",ls="--",c='gray', lw=0.25, alpha=0.25)  
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)    
        return axs
    
    def histogram_plot(self, axs, data, bins, xlim=None, ylim=None, 
        histcolor='gray', xaxislabel='x axis', alpha=1, density=True): 
        """ histogram_plot function
        takes data and makes a histogram
        ================INPUTS============= 
        data is data
        bins are bins
        xlim is x limits; default is None (which computes max/min)
        ylim is y limits; default is None (which lets it keep its default)
        histcolor is histogram colour (default is gray)
        xaxislabel is x axis label (default is 'x axis')
        alpha is histogram transparency (default 1)
        density is if to plot as pdf, default yes
        ================OUTPUT============= 
        axs is axis object """
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if xlim is None:
            xlim = np.array([np.min(data), np.max(data)])

        axs.hist(data, bins=bins, density=density, color=histcolor, alpha=alpha);
        axs.grid(True,which="both",ls="--",c='gray', lw=0.25, alpha=0.25) 
        if density==True:
            axs.set_ylabel('probability density', fontsize=fontsz)
        else:
            axs.set_ylabel('frequency', fontsize=fontsz)    
        axs.set_xlim(xlim)
        if ylim is not None:
            axs.set_ylim(ylim)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        return axs

    def scatter_plot(self, axs, x, y, xlim=None, ylim=None, label='',
        edgecolor='k', facecolor='white', s=5, lw=0.75, xaxislabel='x axis', yaxislabel='y axis', alpha=1):
        """ scatter_plot function
        takes data and makes a scatter plot
        ================INPUTS============= 
        x is x data
        y os y data
        xlim is x limits; default is None (which computes max/min)
        ylim is y limits; default is None (which computes max/min)
        label is label; default is nothing
        edgecolor is edge colour; default is black
        facecolor is face colour; default is white
        s is size of scatter point; default is 5
        lw is line width (default 0.75)
        xaxislabel is x axis label (default is 'x axis')
        yaxislabel is y axis label (default is 'y axis')
        ================OUTPUT============= 
        axs is axis object """
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8
        
        if xlim is None:
            xlim = np.array([np.min(x), np.max(x)])
        if ylim is None:
            ylim = np.array([np.min(y), np.max(y)])
        axs.scatter(x, y, s=s, edgecolors=edgecolor, facecolor=facecolor, lw=lw, label=label, alpha=alpha)
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        axs.grid(True,which="both",ls="--",c='gray', lw=0.25, alpha=0.25)  
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)    
        return axs

    def image_plot(self, axs, data, vmin=None, vmax=None, cmap='binary', cbar='on', cbarlabel='intensity', label='', 
                   labelcolor='black', pixelsize=69, scalebarsize=5000, scalebarlabel=r'5$\,\mu$m', alpha=1):
        """ image_plot function
        takes image data and makes an image plot
        ================INPUTS============= 
        data is image
        vmin is minimum pixel displayed (default 99.9%)
        vmax is maximum pixel displayed (default 0.1%)
        cmap is colour map used; default gray
        cbarlabel is colour bar label; default intensity
        label is any annotation
        labelcolor is annotation colour
        pixelsize is pixel size in nm for scalebar, default 69
        scalebarsize is scalebarsize in nm, default 5000
        scalebarlabel is scale bar label, default 5 um
        ================OUTPUT============= 
        axs is axis object """
        if vmin is None:
            vmin = np.percentile(data.ravel(), 0.1)
        if vmax is None:
            vmax = np.percentile(data.ravel(), 99.9)

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        
        im = axs.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin='lower')
        if cbar == 'on':
        	cbar = plt.colorbar(im, fraction=0.045, pad=0.02, ax=axs, location='left')
        	cbar.set_label(cbarlabel, rotation=90, labelpad=1, fontsize=8)
        	cbar.ax.tick_params(labelsize=7, pad=0.1, width=0.5, length=2) 
        axs.set_xticks([])
        axs.set_yticks([])
        pixvals = scalebarsize/pixelsize
        scalebar = AnchoredSizeBar(axs.transData,
                                        pixvals, scalebarlabel, 'lower right', 
                                        pad=0.1,
                                        color=labelcolor,
                                        frameon=False,
                                        size_vertical=1)

        axs.add_artist(scalebar)
        axs.annotate(label, xy=(5, 5), xytext=(20, 60),
                        xycoords='data', color=labelcolor, fontsize=6)

        return axs
