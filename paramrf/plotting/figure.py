import os

import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
try: 
    import plotly.graph_objs as go
    plotly_available = True
    from plotly.subplots import make_subplots
except ImportError:
    plotly_available = False
    
try:
    from fgivenx import plot_contours, plot_lines
    fgivenx_available = True
except ImportError:
    fgivenx_available = False

try:
    from anesthetic import make_2d_axes, NestedSamples
    anesthetic_available = True
except ImportError:
    anesthetic_available = False

class GridFigure():
    """
    An abstraction over a grid figure (matplotlib or plotly) with convenience functions to make plotting easier and more consistent.
    item_layout example: [{"colspan": 2}, None], [{}, {}]
    """
    def __init__(self, num_items=1, item_layout: list[list] = None, framework = 'matplotlib', interactive = False, fig_size = (8, 6), fig_dpi = 100):
        if framework == 'plotly' and not plotly_available:
            raise Exception('Plotly not available but requested')
        
        if interactive and not framework == 'plotly':
            raise Exception('Interactive plots require <plotly> as the plotting framework')        
        
        self.framework = framework
        self.interactive = interactive
        self.fig_handle = None
        self.fig_size = fig_size
        self.fig_dpi = fig_dpi
        
        self._num_items = num_items
        self._item_layout = item_layout or [[{}]]
        self._trace_map = {}
        self._grid_spec = None
        self._num_item_cols = 1
        self._num_item_rows = 1
        self._row = 0
        self._col = 0
                
        if num_items == 1:
            self._num_item_cols = 1
        else:
            self._num_item_cols = min(int(np.ceil(np.sqrt(num_items))), 4)
        
        self._num_item_rows = int(np.ceil(num_items / self._num_item_cols))

        num_cols = self._num_item_cols * self.cols_per_item
        num_rows = self._num_item_rows * self.rows_per_item
        
        row_tiled = [row * self._num_item_cols for row in item_layout]
        self.spec = row_tiled * self._num_item_rows
        
        if self.framework == 'matplotlib':
            fig_handle = plt.figure()
            self._grid_spec = fig_handle.add_gridspec(num_rows, num_cols)
            fig_handle.set_figwidth(fig_size[0])
            fig_handle.set_figheight(fig_size[1])
        elif self.framework == 'plotly':
            subplot_titles = []
            for row in range(num_rows):
                for item_col_index in range(num_cols):
                    if not self.spec[row][item_col_index] is None:
                        subplot_titles.append(f'{row+1} {item_col_index+1}')
            
            fig_handle = make_subplots(rows=num_rows, cols=num_cols, specs=self.spec, subplot_titles=subplot_titles)
            if interactive:
                fig_handle = go.FigureWidget(fig_handle)
            fig_handle.update_layout(width=fig_size[0]*100, height=fig_size[1]*100, showlegend=False, margin=dict(l=25, r=25, t=50, b=50))
                
        self.fig_handle = fig_handle
        
    @property
    def rows_per_item(self):
        return len(self._item_layout)

    @property
    def cols_per_item(self):
        return len(self._item_layout[0])
    
    @property
    def num_item_rows(self):
        return self._num_item_rows
    
    @property
    def num_item_cols(self):
        return self._num_item_cols
    
    @property
    def num_cols(self):
        return self._num_item_cols * self.cols_per_item
    
    @property
    def num_rows(self):
        return self._num_item_rows * self.rows_per_item
    
    def start(self):
        self._row = 0
        self._col = 0
        return self.current()
    
    def current(self):
        return (self._row, self._col)
    
    def next(self):
        self._col += self.cols_per_item        
        if self._col + 1 > self.num_cols - 1:
            self._col = 0
            self._row += self.rows_per_item        
            
        return self.current()
    
    def get_axis(self, pos):
        if self.framework == 'matplotlib':
            row, col = pos
            spec = self.spec[row][col]
            span = spec.get('colspan', 1)
            if span == 1:
                target_spec = self._grid_spec[row, col]
            else:
                target_spec = self._grid_spec[row, col:(col+span)]
            
            existing_ax = None
            for ax in self.fig_handle.get_axes():
                if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() == target_spec:
                    existing_ax = ax
                    break

            if existing_ax is None:
                ax = self.fig_handle.add_subplot(target_spec)
            else:
                ax = existing_ax            
            
            return ax
        else:
            raise Exception('Axes not returnable for other frameworks')
    
    def plot(self, pos, x, y_vals, labels=None, title=None, ylabel=None, xlabel=None, colors = None, linestyles = None, legend = False):
        if self.framework == 'matplotlib':
            ax = self.get_axis(pos)
            
            if not title is None:
                ax.set_title(title)                
            if not xlabel is None:
                ax.set_xlabel(xlabel)                
            if not ylabel is None:
                ax.set_ylabel(ylabel)            
            for i, y in enumerate(y_vals):
                label = labels[i] if labels else ''
                linestyle = linestyles[i] if linestyles else '-'
                color = colors[i] if colors else 'black'
                ax.plot(x, y, color=color, linestyle=linestyle, label=label)
                
            if labels and legend:
                ax.legend()
        elif self.framework == 'plotly':
            if self.interactive:
                trace_map = self._trace_map
            else:
                trace_map = {}
            
            row, col = pos[0] + 1, pos[1] + 1
            for i, (y, color) in enumerate(zip(y_vals, colors)):
                key = f'{row} {col} {i}'
                if not key in trace_map:
                    self.fig_handle.add_trace(go.Scatter(x=x, y=[], mode="lines", line=dict(color=color)), row=row, col=col)
                    trace_map[key] = self.fig_handle.data[-1]
                
                    self.fig_handle.update_yaxes(title=title, row=row, col=col)
                    self.fig_handle.update_xaxes(title=xlabel, row=row, col=col)
                    self.fig_handle.update_annotations(selector={"text": f"{row} {col}"}, text=title)
            
                trace_map[key].y = y
            
    def plot_contours(self, pos, x, func=None, samples=None, weights=None, title=None, ylabel=None, xlabel=None):
        if self.framework == 'matplotlib':
            ax = self.get_axis(pos)

            if not title is None:
                ax.set_title(title)                
            if not xlabel is None:
                ax.set_xlabel(xlabel)                
            if not ylabel is None:
                ax.set_ylabel(ylabel)                              
            plot_contours(func, x, samples, ax=ax, weights=weights)
            # plot_contours(func, x, samples, ax=ax)
            # plot_lines(func, x, samples, ax=ax, weights=weights)
        else:
            raise Exception('Contours can only be plotted with matplotlib')
        
    def plot_lines(self, pos, x, func=None, samples=None, weights=None, title=None, ylabel=None, xlabel=None):
        if self.framework == 'matplotlib':
            ax = self.get_axis(pos)

            if not title is None:
                ax.set_title(title)                
            if not xlabel is None:
                ax.set_xlabel(xlabel)                
            if not ylabel is None:
                ax.set_ylabel(ylabel)                              
            plot_lines(func, x, samples, ax=ax, weights=weights, color='red')
        else:
            raise Exception('Lines can only be plotted with matplotlib')        
        
    def save(self, output_prefix, title):
        if output_prefix is None:
            output_prefix = ''
        else:
            output_prefix += '/'
        
        if self.framework == 'matplotlib':
            os.makedirs(output_prefix, exist_ok=True)
            self.fig_handle.tight_layout()
            self.fig_handle.savefig(f'{output_prefix}{title}.png', dpi=self.fig_dpi)
        elif self.framework == 'plotly':
            self.fig_handle.write_image(f'{output_prefix}{title}.png')
            
    def close(self):
        if self.framework == 'matplotlib':
            plt.close(self.fig_handle)
        elif self.framework == 'plotly':
            pass
    
    def tight_layout(self):
        if self.framework == 'matplotlib':
            self.fig_handle.tight_layout()
        elif self.framework == 'plotly':
            pass            
        
    def legend(self, loc=None):
        if self.framework == 'matplotlib':
            fig = self.fig_handle
            
            # Remove duplicates
            handles, labels = [], []
            for ax in fig.axes:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)

            # Remove duplicates using a dict
            by_label = dict(zip(labels, handles))

            # Create legend on the figure level
            fig.legend(by_label.values(), by_label.keys(), loc=loc)
        elif self.framework == 'plotly':
            pass