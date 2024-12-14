# deploy statment: rsconnect deploy shiny C:\Users\Josiah\Desktop\data_projects\cpi_dashboard --name iu-msds-josiahkeime --title CPI_dashboard


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from shiny.express import ui, input, render
from shinywidgets import render_plotly, render_widget
import plotly.express as px
from shiny import reactive, render
from faicons import icon_svg
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from shiny.ui import page_navbar
import torch

# Load clean data
infile = Path(__file__).parent / "data_cleaned.csv"
df_clean = pd.read_csv(infile)

# helper function to create df from a column and CPI values
def column2df(df, column, col_values):
  dict_cat_corr = {}
  for c in col_values:
    dict_cat_corr[c] = df['CPI'][df[column]== c].rename(c)
    dict_cat_corr[c].reset_index(drop=True,inplace=True)
  return pd.concat(dict_cat_corr.values(),axis=1)

# dataframe for PCA analysis
cat_pca_df = column2df(df_clean, 'Category',  ['Transportation','Housing','Energy','Food and beverages', 'Medical care', 'Education'])

# Loadding model for Anomly detection
class AnomlyDetector_model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # create encoder model
    self.encoder = torch.nn.Sequential( torch.nn.Linear(120,60), torch.nn.ReLU(),torch.nn.Linear(60,30))
    # create decoder model
    self.decoder = torch.nn.Sequential(torch.nn.Linear(30,60), torch.nn.ReLU(), torch.nn.Linear(60,120))
  # create function for passing data through the model for training
  def forward(self,x):
    self.train()
    encode = self.encoder(x)
    decode = self.decoder(encode)
    return decode
  # create function for passing data through the model for predictions
  def predict(self,x):
    self.eval()
    encode = self.encoder(x)
    decode = self.decoder(encode)
    return decode.detach()

AE_model = AnomlyDetector_model()
model_file = Path(__file__).parent / "d_model.pth"
AE_model.load_state_dict(torch.load(model_file))


ui.page_opts(
    title="CPI Dashboard", id="page")

################################"About Page"###########################################################
css_file = Path(__file__).parent / "css" / "styles.css"

with ui.nav_panel("About"):
    ui.head_content(ui.include_css(css_file))
    ui.HTML(
        """
        <div class="main-container">
            <!-- Main content -->
            <header class="header bg-primary-subtle lead">
                <H1>U.S. Consumer Price Index (CPI)</H1>  
                <H1>Analysis Dashboard</H1>  


            </header>
            
            <div class="content">
                <H4>About</H4>
                <p>
                    This web app is designed to analyze and predict trends in U.S. Consumer Price Index (CPI)
                    from data provided by the U.S. Bureau of Labor Statistics (BLS). This tool allows users to
                    gain insights into U.S. economic patterns and make informed decisions based on historical
                    and predictive analytics.
                </p>
                <br>
                <H4>What is CPI?</H4>
                <p>
                    "The Consumer Price Index (CPI) is a measure of average change over time in the prices paid by urban consumers for a
                    market basket of consumer goods and services. The CPI measures inflation as experienced by consumers in their day-to-day living expenses. The CPI represents all goods and services purchased for consumption by the reference population. BLS has classified all expenditure items into more than 200 categories, arranged into eight major groups (food and beverages, housing, apparel, transportation, medical care, recreation, education and communication, and other goods and services). ” (U.S. Bureau of Labor Statistics, n.d.).
                </p>
                <br>
                <H4>App Features</H4>
                <ul>
                    <li>Visualize and explore CPI trends based on various categories and geographical regions.</li>
                    <li>Analyze the data by examining correlations, principal components, seasonal decompositions, and anomalies.</li>
                    <li>Forecast future trends by using machine learning algorithms trained on historical data.</li>
                </ul>
            </div>
            <!-- Footer -->
            <footer class="footer">
                <p>DSCI D590 Applied Data Science Fall 2024 - Final Project, Part 3</p>
                <p>Web app created by Leona Chase, Jing Hao, and Josiah Keime</p>

            </footer>
        </div>
        """
    )


################################"Explorer Page"###########################################################

with ui.nav_panel("Data Explorer"):    
    with ui.layout_column_wrap(fill=False):
        with ui.value_box(showcase=icon_svg("cart-shopping"), style="padding: 5px; margin: 5px; height: 110px;"):
            "Average CPI"
            
            @render.text
            def CPI_mean():
                return f"{filtered_df()['CPI'].mean():.1f}"
        
        with ui.value_box(showcase=icon_svg("arrow-trend-down"), style="padding: 5px; margin: 5px; height: 110px;"):
            "Minimum CPI"

            @render.text
            def CPI_min():
                min_row = filtered_df().loc[filtered_df()['CPI'].idxmin()]
                # Extract the minimum CPI and the corresponding date
                min_cpi = min_row['CPI']
                return f"{min_cpi:.1f}"

            @render.text
            def min_date():
                min_row = filtered_df().loc[filtered_df()['CPI'].idxmin()]
                min_date = pd.to_datetime(min_row['Date']).strftime('%b %d, %Y')             
                return f"on {min_date}"
            
        with ui.value_box(showcase=icon_svg("arrow-trend-up"), style="padding: 5px; margin: 5px; height: 110px;"):
            "Maximum CPI"

            @render.text
            def CPI_max():
                max_row = filtered_df().loc[filtered_df()['CPI'].idxmax()]
                # Extract the maximum CPI and the corresponding date
                max_cpi = max_row['CPI']
                return f"{max_cpi:.1f}"

            @render.text
            def max_date():
                max_row = filtered_df().loc[filtered_df()['CPI'].idxmax()]
                max_date = pd.to_datetime(max_row['Date']).strftime('%b %d, %Y')
                
                return f"on {max_date}"
    with ui.card(full_screen=True):        
        with ui.layout_sidebar():
            with ui.sidebar(width = 680, title="Filter controls"):
                with ui.card():
                    with ui.layout_columns(col_widths=[3, 6, 3]):
                        ui.input_checkbox_group(
                            "Year",
                            ui.HTML('<b>Year</b>'),
                            df_clean['Year'].unique().tolist(),
                            selected=df_clean['Year'].unique().tolist(),
                        )                    

                        ui.input_checkbox_group(
                            "Category",
                            ui.HTML('<b>Category</b>'),
                            df_clean['Category'].unique().tolist(),
                            selected=df_clean['Category'].unique().tolist(),
                        )
                        ui.input_checkbox_group(
                            "Region",
                            ui.HTML('<b>Region</b>'),
                            df_clean['Region'].unique().tolist(),
                            selected=df_clean['Region'].unique().tolist(),
                        )

            with ui.navset_tab():      

                with ui.nav_panel("Plot: Average U.S. CPI by Category"):
                    @render_widget  
                    def categorical_cpi():  
                        scatterplot = px.scatter(
                            categorical_filtered_df()[categorical_filtered_df()['Region'] == 'US'],
                            x='Date',
                            y='CPI',
                            color='Category',  
                            title='Average U.S. CPI by Category ',
                        ).update_layout(
                            xaxis_title='Date',
                            yaxis_title='CPI',
                            legend_title='Category',
                            xaxis=dict(tickangle=45),
                            yaxis=dict(range=[100,600]),  
                            template='plotly_white'
                        )

                        return scatterplot  
            
                with ui.nav_panel("Plot: Average CPI for All Items by Region"):
                    @render_widget  
                    def regional_cpi():  
                        scatterplot = px.scatter(
                            regional_filtered_df()[regional_filtered_df()['Category'] == 'All items'],
                            x='Date',
                            y='CPI',
                            color='Region',  
                            title='Average CPI for All Items in the U.S. by Region',
                        ).update_layout(
                            xaxis_title='Date',
                            yaxis_title='CPI',
                            legend_title='Region',
                            xaxis=dict(tickangle=45),
                            yaxis=dict(range=[100,600]),  
                            template='plotly_white'
                        )

                        return scatterplot  

                with ui.nav_panel("Table: CPI Data"):
                    @render.data_frame
                    def summary_statistics():
                        cols = [
                            "Year",
                            "Category",
                            "Region",
                            "Month",
                            "CPI",
                            "Date"
                        ]
                        return render.DataGrid(filtered_df()[cols], filters=True)
                    
@reactive.calc
def filtered_df():
    filt_df = df_clean
    filt_df = filt_df[filt_df["Year"].astype(str).isin(input.Year())]
    filt_df = filt_df[filt_df["Region"].isin(input.Region())]
    filt_df = filt_df[filt_df["Category"].isin(input.Category())]

    return filt_df

@reactive.calc
def regional_filtered_df():
    filt_df = df_clean
    filt_df = filt_df[filt_df["Year"].astype(str).isin(input.Year())]
    filt_df = filt_df[filt_df["Region"].isin(input.Region())]

    return filt_df

@reactive.calc
def categorical_filtered_df():
    filt_df = df_clean
    filt_df = filt_df[filt_df["Year"].astype(str).isin(input.Year())]
    filt_df = filt_df[filt_df["Category"].isin(input.Category())]

    return filt_df

################################"Analysis Page "###########################################################


with ui.nav_panel("Analysis"):  
    with ui.navset_card_underline():
#### Correlation Panel ##############################        
        with ui.nav_panel("Correlation"):
            with ui.layout_sidebar():
                
                with ui.sidebar(title="Filter controls"):
                    ui.input_checkbox_group(
                        "cat_heatmap",
                        "CPI Categories",
                        df_clean['Category'].unique().tolist(),
                        selected=df_clean['Category'].unique().tolist(),
                    )
                  
######### Correlation Matrix ###############
                with ui.card(full_screen=True):
                    ui.card_header("CPI Categories Correlation Matrix ") 
                    
                    @render_widget  
                    def categorical_heatmap():  
                        heatmap = px.imshow(categorical_corr_matrix(), text_auto=True, aspect="auto")
                        return heatmap
                    
                    @reactive.calc
                    def categorical_corr_matrix():
                        dict_cat_corr = {}
                        for c in input.cat_heatmap():
                            dict_cat_corr[c] = pd.Series(df_clean['CPI'][df_clean['Category']==c], name= c)
                            dict_cat_corr[c].reset_index(drop=True,inplace=True)
                        df_corMatrix = pd.concat(dict_cat_corr,axis=1)
                        return df_corMatrix.corr()
#### PCA Panel ##############################
        with ui.nav_panel("PCA"):
            @render_widget  
            def scree_plot():  
                scree = px.line(cum_sum_explain_ratio()).update_layout(xaxis_title="Number of Components"
                                                                       ,yaxis_title="Cumulative Explained Ratio"
                                                                      ,showlegend=False
                                                                       ,title='CPI Category Scree Plot'
                                                                       , title_xanchor='auto')
                return scree

            @reactive.calc
            def cum_sum_explain_ratio():
                scaler_screen = StandardScaler()
                pca_screener = PCA()
                pca_screen_pipeline = make_pipeline(scaler_screen, pca_screener)
                pca_screen_pipeline.fit(cat_pca_df);
                return np.cumsum(pca_screener.explained_variance_ratio_)


            (ui.input_slider("select_comp", "Select the number of components", 1, 6, 2),)  
            @render.text
            def value():
                return f"{input.select_comp()}"

            @reactive.calc
            def pca_analysis():
                n = input.select_comp()
                cols = []
                for i in range(1,n+1):
                    cols.append('PC'+str(i))

                scaler = StandardScaler()
                pca = PCA(n_components=i)
                pca_pipeline = make_pipeline(scaler, pca)
                pca_pipeline.fit(cat_pca_df)
                pca_a_df = pd.DataFrame(pca.components_.transpose(),index = cat_pca_df.columns, columns = cols)
                return pca_a_df.round(decimals=3)

            @render_widget  
            def pca_heatmap():  
                pca_map = px.imshow(pca_analysis(), text_auto=True, aspect="auto").update_layout( title='CPI Cateogory Principal Components'
                                                                       , title_xanchor='auto')
                return pca_map
                        
#### Seasonal Panel ##############################

        with ui.nav_panel("Seasonality"):
            
            # SETUP FOR THE SEASONALITY PANEL
            # Create new dataframe for monthly CPI means in the US for all items...
            df_allUS = df_clean[(df_clean['Category'] == 'All items') & (df_clean['Region'] == 'US')]
            df_allUS = df_allUS[['Date','CPI']]
            df_allUS['Date'] = pd.to_datetime(df_allUS['Date'])
            df_allUS = df_allUS.set_index('Date')
            monthly_CPI = df_allUS.resample('ME').mean() 

            # Handle non-stationarity for monthly CPI...

            # Take the first difference to remove trends 
            monthly_CPI_diff = monthly_CPI.diff().dropna()
            # Apply seasonal difference to remove seasonality, previous data analysis indicated seasonality trends at every 6 months 
            monthly_CPI_seasonal_diff = monthly_CPI_diff.diff(6).dropna()

            
            ui.HTML(
                """
                <div style="margin: 20px auto;">
                    <H4 style="text-align: center; font-weight: bold;">
                    Seasonality Analysis for Average Monthly CPI in the U.S. for All Items 
                    </H4>
                </div>
                """
            )
            with ui.layout_column_wrap(fill=False):
                with ui.card(full_screen=True):  
                    @render.plot
                    def seas_decomp():          
                        series = monthly_CPI.squeeze("columns")
                        result = seasonal_decompose(series, model='additive')

                        fig = result.plot()
                        fig.suptitle("Seasonal Decomposition ")
                        fig.tight_layout()
                        return fig

                    
                with ui.card(full_screen=True):    
                    @render.plot
                    def ACF_plot():          
                        fig, ax = plt.subplots()
                        plot_acf(monthly_CPI_seasonal_diff, ax=ax)
                        ax.set_title("Autocorrelation Function (ACF) ")
                        fig.tight_layout()
                        return fig
                    
                with ui.card(full_screen=True):
                    @render.plot
                    def PACF_plot():          
                        fig, ax = plt.subplots()
                        plot_pacf(monthly_CPI_seasonal_diff, ax=ax)
                        ax.set_title("Partial Autocorrelation Function (PACF)")
                        return fig       
#### Anomaly Panel ##############################
        with ui.nav_panel("Anomaly Detect"):
            @reactive.calc
            def get_anomaly():
                val = df_clean.query('Category == "All items" and Region == "US"')[['CPI','Date']].sort_values(by='Date')
                reconstruct = AE_model.predict(torch.from_numpy(val['CPI'].values).type(torch.float32))
                orgin = val['CPI'].values
                reconstruct = reconstruct.detach().numpy()
                anomaly_scores = np.sqrt(np.abs(reconstruct - orgin))
                threshold = np.quantile(anomaly_scores, input.select_percent())
                anomalous = anomaly_scores > threshold
                val['anomaly'] = anomalous
                print(type(val))
                return val

            (ui.input_slider("select_percent", "Select Anomaly Threshold", 0, 1, .9),)  
            @render.text
            def thr_percent():
                return f"{input.select_percent()}"
            
            @render_widget  
            def anomaly_graph():  
                print(type(get_anomaly()))
                an_plot = px.scatter(get_anomaly(),x="Date", y="CPI", color="anomaly",symbol='anomaly'
                                )#.set(title="All Items CPI Anomalies");
                return an_plot

################################"Predictions Page "###########################################################


with ui.nav_panel("Forecasting Models"):
    with ui.navset_card_underline():
################################ "Model Comparison Panel "###########################################################       
        with ui.nav_panel("Average CPI for All Items in the U.S."):      
            # SETUP FOR SARIMA MODEL 
            # SARIMA Parameters based on Tuning done in Part 2
            p, d, q = 1,1,1
            P, D, Q, s = 1, 1, 1, 6   

            # Split data to training/ test sets
            train_size = int(len(monthly_CPI) * 0.8)  # Use 80% of data for training
            train_data = monthly_CPI[:train_size]
            test_data = monthly_CPI[train_size:]

            sarima =  SARIMAX(monthly_CPI, order=(p, d, q), seasonal_order=(P, D, Q, s))
            fitted_sarima = sarima.fit()
            # Set forecast steps to 60 months (5 years) in the future
            forecast_steps = 60

            # Forecast future values
            forecast = fitted_sarima.get_forecast(steps=forecast_steps)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()

            forecast_dates = pd.date_range(start=monthly_CPI.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='ME')
            forecast_df = pd.DataFrame(forecast_mean.values, index=forecast_dates, columns=['Forecast'])

            with ui.card(full_screen=True):
                @render_widget  
                def sarima_forecast():  

                    # Create the graph
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=monthly_CPI.index, y=monthly_CPI['CPI'], mode='lines', name='Observed'))
                    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast'))

                    # Plot confidence intervals
                    ci_df = pd.DataFrame({
                        'Date': forecast_ci.index,
                        'Lower Bound': forecast_ci.iloc[:, 0],
                        'Upper Bound': forecast_ci.iloc[:, 1]
                    })

                    fig.add_traces([
                        go.Scatter(
                            x=ci_df['Date'],y=ci_df['Lower Bound'],mode='lines', name='Lower Confidence Interval',
                            fill='tonexty',fillcolor='rgba(255, 192, 203, 0.3)',line=dict(color='rgba(255, 192, 203, 0)')
                        ),
                        go.Scatter(
                            x=ci_df['Date'], y=ci_df['Upper Bound'],mode='lines', name='Upper Confidence Interval',
                            fill='tonexty',fillcolor='rgba(255, 192, 203, 0.3)',line=dict(color='rgba(255, 192, 203, 0)')
                        )
                    ])

                    fig.update_layout(title='SARIMA 5 Year Forecast for Average CPI of All Items in the U.S.', xaxis_title='Date', yaxis_title='CPI')
                    
                    return fig 

 ################################ Forecasting Models Panel ###########################################################  
        with ui.nav_panel("Average Housing CPI in the U.S."):
            with ui.layout_column_wrap(fill=False):
                with ui.card(full_screen=True):  
                    @render.plot
                    def forecast_plot():
                        # Step 1: Filter the data for the "Housing" category
                        df_housing = filtered_df()[filtered_df()['Category'] == "Housing"]

                        # Ensure 'Year' is numeric and CPI values are available
                        df_housing['Year'] = pd.to_numeric(df_housing['Year'], errors='coerce')

                        # Step 2: Aggregate by Year and calculate the mean CPI for each year (if multiple months)
                        housing_data = df_housing.groupby('Year')['CPI'].mean()

                        # Step 3: Fit an ARIMA model to the annual average data
                        arima_model = ARIMA(housing_data, order=(1, 1, 0))  # ARIMA(p=1, d=1, q=0)
                        arima_model_fit = arima_model.fit()

                        # Step 4: Make forecast for the next 5 years
                        forecast_steps = 5  # Forecast next 5 years
                        arima_forecast_values = arima_model_fit.forecast(steps=forecast_steps)

                        # Step 5: Fit Simple Exponential Smoothing (SES)
                        ses_model = SimpleExpSmoothing(housing_data)
                        ses_model_fit = ses_model.fit()

                        # SES Forecast for future values
                        ses_forecast_values = ses_model_fit.forecast(steps=forecast_steps)

                        # Plotting both ARIMA and SES forecasts together
                        forecast_years = list(range(housing_data.index.max() + 1, housing_data.index.max() + 1 + forecast_steps))  # Future years
                        
                        # Create the plot
                        fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes
                        
                        # Historical Data
                        ax.plot(housing_data.index, housing_data, label="Historical Data", color="green", marker='o')
                        
                        # ARIMA Forecast
                        ax.plot(forecast_years, arima_forecast_values, label="ARIMA Forecast", color="red", marker='o')
                        
                        # SES Forecast
                        ax.plot(forecast_years, ses_forecast_values, label="SES Forecast", color="blue", marker='o')
                        
                        ax.set_title("ARIMA and SES Forecast for Housing CPI in the U.S.")
                        ax.set_xlabel("Year")
                        ax.set_ylabel("Average CPI")
                        ax.legend(loc="upper left")
                        ax.grid(True)
                        fig.tight_layout()  # Adjust layout to fit everything
                        
                        return fig


################################ "Holt-Winters" ###########################################################                      
                with ui.card(full_screen=True):             
                    @render.plot
                    def holt_winters_forecast_plot():
                        # Step 1: Filter the data for the "Housing" category (or any other category)
                        df_housing = filtered_df()[filtered_df()['Category'] == "Housing"]

                        # Ensure 'Date' is a datetime object and 'CPI' is numeric
                        df_housing['Date'] = pd.to_datetime(df_housing['Date'], errors='coerce')
                        df_housing = df_housing.dropna(subset=['CPI'])  # Drop any rows with NaN CPI values

                        # Step 2: Set the date column as the index for time series forecasting
                        df_housing.set_index('Date', inplace=True)

                        # Step 3: Resample to monthly data if not already monthly
                        df_monthly = df_housing['CPI'].resample('M').mean()  # Monthly average CPI

                        # Step 4: Fit the Holt-Winters Exponential Smoothing model
                        hw_model = ExponentialSmoothing(df_monthly, trend='add', seasonal='add', seasonal_periods=12)
                        hw_model_fit = hw_model.fit()

                        # Step 5: Make forecast for the next 12 months
                        forecast_steps = 12  # Forecast next 12 months
                        hw_forecast = hw_model_fit.forecast(steps=forecast_steps)

                        # Step 6: Create the plot
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Plot the historical data
                        ax.plot(df_monthly.index, df_monthly, label="Historical Data", color="green", marker='o')

                        # Plot the Holt-Winters forecast
                        forecast_index = pd.date_range(df_monthly.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='M')
                        ax.plot(forecast_index, hw_forecast, label="Holt-Winters Forecast", color="blue", marker='o')

                        ax.set_title("Holt-Winters Forecast for Housing CPI")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("CPI")
                        ax.legend(loc="upper left")
                        ax.grid(True)
                        fig.tight_layout()

                        return fig




