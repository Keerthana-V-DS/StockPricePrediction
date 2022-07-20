import re
import matplotlib.pyplot as plt 
from PIL import Image
import base64, spacy, requests, streamlit as st, numpy as np, pandas as pd, yfinance as yf 
#from fbprophet import Prophet
from prophet import Prophet
#from fbprophet.plot import plot_plotly, plot_components_plotly
from prophet import plot

from datetime import timedelta
from bs4 import BeautifulSoup
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import  datetime, timedelta
from metrics import RMSE, MAE, MAPE, MSE, SMAPE, confusion_matrix
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from GoogleNews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import spacy
nltk.download('vader_lexicon')



def main():
    # global tickers
    # tickers = ['AMZN','AAPL','TCS.NS','GOOG','ITC.NS','IDEA.NS','BPCL.NS', 'ONGC.NS', 'RDS-A', 'RDS-B','INFY.NS', 'ABBOTINDIA.NS']
    df_ticker = pd.read_csv('data/ind_nifty500list.csv')
    df_ticker['Symbol'] = df_ticker['Symbol']+'.NS'
    tickers = df_ticker['Symbol'].values
    
    # Page config
    st.set_page_config(page_title="Forecast")

    agenda = ["Introduction", 'News Feed', 'Data Exploration', 'Sentiment Analysis','Market Movement', 'ARIMA Forecast','SARIMAX Forecast', 'Prophet Forecast']
    choice = st.sidebar.selectbox("Select Activities", agenda)
    components.html("""
        <div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
                {"symbols": [{"proName": "FOREXCOM:SPXUSD","title": "S&P 500"},{"proName": "FOREXCOM:NSXUSD","title": "US 100"},
                            {"proName": "FX_IDC:EURUSD","title": "EUR/USD"},{"description": "SENSEX","proName": "BSE:SENSEX"},
                            {"description": "Amazon","proName": "AMZN"},
                            {"description": "Apple Inc","proName": "AAPL"}, {"description": "TCS","proName": "TCS"},
                            {"description": "Alaphebet Inc","proName": "GOOG"}, {"description": "ITC Inc","proName": "ITC"},
                            {"description": "Vodafone Idea", "proName": "IDEA"}, {"description": "Bharat Petroleum","proName": "BPCL"}, 
                            {"description": "Oil & Natural Gas","proName": "ONGC"}, {"description": "Shell","proName": "RDSA"}, 
                            {"description": "INFOSYS Ltd","proName": "INFY"}],
                "showSymbolLogo": true, "colorTheme": "dark",  "isTransparent": false, "displayMode": "adaptive", "locale": "in"
                }
            </script>
        </div>
    """)
    #st.success('This is a success message!')

    def extract_text(url):
        headings = []
        description = []
        r = requests.get(url)
        soup = BeautifulSoup(r.content, features='lxml')
        headings = soup.findAll('title')
        description = soup.findAll('description')
        return headings, description

    def stock_info(headings, description, flag):
        for title,desc in zip(headings, description):
            doc = nlp(title.text)
            for ent in doc.ents:
                try:
                    if not df[df['Symbol'].str.lower() == ent.text.lower()].empty:
                        symbol = df[df['Company Name'].str.contains(ent.text)]['Symbol'].values[0]
                        org_name = df[df['Company Name'].str.contains(ent.text)]['Company Name'].values[0]
                        html = spacy.displacy.render(doc, style="ent", options={"ents": ['ORG']})
                        html = html.replace("\n", " ")
                        title_col, analyze_col = st.columns([9.5,0.5])
                        with title_col:
                            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
                        with analyze_col:    
                            st.checkbox('Analyze', key=symbol)
                        if flag == 'Economic Times Feed':
                            st.markdown("* " + desc.text.split('</a>')[-1])
                        elif flag == 'Money Control Feed':
                            st.markdown("* " + desc.text.split('/>')[1])                            
                except:
                    pass

    if choice == 'Introduction':
        # page_bg_img = '''
        # <style>
        #     body {
        #         background-image: url("https://miro.medium.com/max/1400/1*T9VUDALam3DIS0wHDWrxBg.png");
        #         background-size: cover;
        #     }
        # </style>
        # '''
        # st.markdown(page_bg_img, unsafe_allow_html=True)
        # st.markdown('<style>body{background-image: url("https://miro.medium.com/max/1400/1*T9VUDALam3DIS0wHDWrxBg.png");background-size: cover;}</style>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Introduction</h3>", unsafe_allow_html=True)
        st.write(
            "\nPredicting Stock Price of a Company in near future has grown complex due to the effect of online social media and news platforms / channels. Existing research on Stock Price prediction is based on one of the following methods.\n"
            "\n * Technical Analysis using Chart Pattern Technical Indicators.\n"
            "\n * Analyzing past stock prices and predicting by Time series models.\n"
            "\n * Analyzing news/Social media Text messages to do sentiment analysis using NLP models.\n"
            "\nThough the first approach is used for a very long time, few aspects are missed e,g, Impact due to public sentiment, New Competition entering market   and misinterpretation of charts etc.\n"
            "\nPredicting Stock market using Timeseries model and sentiment analysis mentioned provide relatively accurate and more informed price prediction based on the history of data and external factors that are affecting the public view on the company.\n"
        )
        st.markdown("![Alt Text](https://i.gifer.com/7JbT.gif)") 

    if choice == 'News Feed':
        st.markdown("<h3 style='text-align: center;'>Latest News!</h3>", unsafe_allow_html=True)
        # st.markdown("![Alt Text](https://i.gifer.com/7D7o.gif)")
        HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
        try:
            nlp = spacy.load("en_core_web_sm")
        except:  # If not present, we download
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        df = pd.read_csv('data/ind_nifty500list.csv')
        st.header('Stock Universe')
        with st.expander("Nifty 500 Tickers", expanded=False):
            st.dataframe(df[['Company Name','Industry','Symbol']].set_index('Symbol'))
        st.header('News Feed')

        with open("main.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        feed_source = st.radio('', options=['Economic Times Feed','Money Control Feed'] )
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        
        if feed_source == 'Economic Times Feed':
            url = 'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms'
        elif feed_source == 'Money Control Feed':
            url = 'https://www.moneycontrol.com/rss/technicals.xml'

        headings, description = extract_text(url)
        stock_info(headings[3:], description[3:], flag=feed_source)
        for k,v in st.session_state.items():
            if v:
                st.write(k,v)

    if choice == 'Data Exploration':
        st.markdown("<h3 style='text-align: center;'>Explore the Stocks</h3>", unsafe_allow_html=True)
        df = pd.read_csv('data/ind_nifty500list.csv')
        df['Symbol'] = df['Symbol']+'.NS'
        # tick = st.text_input('Add tickers in the list')
        # print(tick)
        #dropdown = st.sidebar.selectbox('Pick your assets', df['Symbol'].values)
        st.markdown("<h1 style='text-align: center;'>Financial Dashboard</h1>", unsafe_allow_html=True)
        start = st.sidebar.date_input('Start',  value=pd.to_datetime('2021-01-01'))
        end = st.sidebar.date_input('End',  value=pd.to_datetime('today'))
        # Initially added to select a ticker on the fly - changed it back to top 500 Nifty
            #dropdown = st.sidebar.selectbox('Pick your assets', tickers)
            # tick = st.sidebar.text_input('New ticker')
            # global tickers 
            # st.session_state.key = tick
            # tickers.append(str(tick))
            # tickers.append(st.session_state.key)
        dropdown = st.sidebar.selectbox('Pick your assets', tickers)

        if len(dropdown):
            stock_info = yf.Ticker(dropdown).info
            logo_col, title_col = st.columns([0.5,5])
            logo_col.image(stock_info['logo_url'])
            title_col.header(stock_info['longName'])
            summary = st.expander(label='Summary', expanded=False)
            with summary:
                st.write(stock_info['longBusinessSummary'])        
            df = yf.download(dropdown, start, end).reset_index()
            df['Date'] = df['Date'].dt.date
            fig = go.Figure(data=[go.Candlestick(x=df['Date'],open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'])])
            fig.update_xaxes(rangeslider_visible=True, 
                rangeselector=dict(buttons=list([dict(count=1, label="1y", step="year", stepmode="backward"),
                                                dict(count=2, label="3y", step="year", stepmode="backward"),
                                                dict(count=3, label="5y", step="year", stepmode="backward"),
                                                dict(step="all")])))
            chart = st.expander(label='Chart', expanded=False)
            with chart:
                st.plotly_chart(fig)          
            fun_metrics = ['ebitdaMargins','profitMargins','grossMargins','operatingCashflow',
                            'revenueGrowth','operatingMargins','ebitda','currentPrice','earningsGrowth',
                            'debtToEquity','totalCash','totalDebt','totalRevenue','totalCashPerShare',
                            'revenuePerShare','priceToBook','forwardPE','marketCap']
            fun_values = [stock_info[metric] for metric in fun_metrics]        
            stock_f = pd.DataFrame()
            stock_f['metric'] = fun_metrics
            stock_f['values'] = fun_values

            def conv_in_million():
                pass
            def conv(value):
                if value > 1000000:
                    return str(round((value/1000000))) + ' M'
                else:
                    return str(value)
        
            stock_f['values'] = stock_f['values'].map(conv)
            
            data_tab = st.expander(label='Stock Price Data', expanded=False)
            fund_tab = st.expander(label='Fundamental Data', expanded=False)
            with data_tab:
                st.dataframe(df.sort_values('Date', ascending=False).set_index('Date'))
            with fund_tab:
                st.table(stock_f.set_index('metric'))        
    
    if choice == 'Sentiment Analysis':
        st.markdown("<h3 style='text-align: center;'>Market Sentiments</h3>", unsafe_allow_html=True)
        dropdown = st.selectbox('Pick your assets', tickers)

        if len(dropdown):
            googlenews = GoogleNews(lang='en',region='IN')
            googlenews.set_period('15d')
            googlenews.search(dropdown)

            result = googlenews.result()
            df = pd.DataFrame(result)
            df = df[['title','media','desc']]

            #st.dataframe(df)
            for i in df['desc'].values:
                st.write(i)

            #Sentiment Analysis
            def percentage(part,whole):
                return 100 * float(part)/float(whole)

            #Assigning Initial Values
            positive = 0
            negative = 0
            neutral = 0
            
            #Creating empty lists
            news_list = []
            neutral_list = []
            negative_list = []
            positive_list = []

            #Iterating over the tweets in the dataframe
            for news in df['desc']:
                news_list.append(news)
                analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
                neg = analyzer['neg']
                neu = analyzer['neu']
                pos = analyzer['pos']
                comp = analyzer['compound']

                if neg > pos:
                    negative_list.append(news) #appending the news that satisfies this condition
                    negative += 1 #increasing the count by 1
                elif pos > neg:
                    positive_list.append(news) #appending the news that satisfies this condition
                    positive += 1 #increasing the count by 1
                elif pos == neg:
                    neutral_list.append(news) #appending the news that satisfies this condition
                    neutral += 1 #increasing the count by 1 

            positive = percentage(positive, len(df)) #percentage is the function defined above
            negative = percentage(negative, len(df))
            neutral = percentage(neutral, len(df))

            #Converting lists to pandas dataframe
            news_list = pd.DataFrame(news_list)
            neutral_list = pd.DataFrame(neutral_list)
            negative_list = pd.DataFrame(negative_list)
            positive_list = pd.DataFrame(positive_list)
            #using len(length) function for counting
            print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n')
            print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n')
            print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n')

         

            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.success(f'Positive Sentiment : %{len(positive_list)*10}')
            with col_2:
                st.info(f'Neutral Sentiment : %{len(neutral_list)*10}')
            with col_3:
                st.error(f'Negative Sentiment: %{len(negative_list)*10}')
                

    if choice == 'Market Movement':
        st.markdown("<h3 style='text-align: center;'>Market Movement</h3>", unsafe_allow_html=True)
        dropdown = st.selectbox('Pick your assets', tickers)

        start = st.date_input('Start',  value=pd.to_datetime('2016-01-01'))
        if len(dropdown):
            df = yf.download(dropdown, start).reset_index()
            df = df[['Date','Adj Close','Volume']].rename(columns = {'Adj Close':'Today'})
            df['Today'] = df['Today'].pct_change()*100
            df['Volume'] = df.Volume.shift(1).values/1000_000_000

            print(df)
            # lag values
            for i in range(1,6):
                df['Lag '+str(i)] = df.Today.shift(i)

            df = df.dropna()
            df['Direction'] = [1 if i>0 else 0 for i in df['Today']]

            df = sm.add_constant(df)
            
            X = df[['const','Lag 1','Lag 2','Lag 3','Lag 4','Lag 5','Volume']]
            y = df.Direction
            st.subheader("Price Movement - Classification Model")
            model = sm.Logit(y,X)

            result = model.fit()
            st.write(result.summary())
            prediction = result.predict(X)

            st.write("")
            st.subheader('Confusion Matrix')
            st.dataframe(confusion_matrix(y, prediction))

            movement = np.sum(np.diag(confusion_matrix(y, prediction)))/df.shape[0]

            # df = df.dropna()
            # X_train = df[df.Date.dt.year < 2021][['const','Lag 1','Lag 2','Lag 3','Lag 4','Lag 5','Volume']]
            # y_train = df[df.Date.dt.year < 2021]['Direction']

            # X_test = df[df.Date.dt.year >= 2021][['const','Lag 1','Lag 2','Lag 3','Lag 4','Lag 5','Volume']]
            # y_test = df[df.Date.dt.year >= 2021]['Direction']

            # st.subheader("Price Movement - Classification Model")

            # model = sm.Logit(y_train, X_train)
            # result = model.fit()

            # st.write(result.summary())
            # prediction = result.predict(X_test)

            # st.write("")
            # st.subheader('Confusion Matrix')
            # st.dataframe(confusion_matrix(y_test, prediction))

            # movement = np.sum(np.diag(confusion_matrix(y_test, prediction)))/df.shape[0]

            if movement > 0.5:
                st.success(f'{movement}')
            else:
                st.error(f'{movement}')
            
    if choice == 'ARIMA Forecast':
        st.markdown("<h3 style='text-align: center;'>ARIMA Forecast</h3>", unsafe_allow_html=True)
        # tickers = ('AMZN','AAPL','TCS.NS','GOOG','ITC.NS','IDEA.NS','BPCL.NS', 'ONGC.NS', 'RDS-A', 'RDS-B','INFY.NS')
        dropdown = st.selectbox('Pick your assets', tickers)

        start = st.date_input('Start',  value=pd.to_datetime('2021-01-01'))
        if len(dropdown):
            df = yf.download(dropdown, start)
            df = df[['Close','Adj Close']]
            df_diff = df.diff(periods=1).dropna().reset_index()
            
            train_end=datetime(2021,12,31)
            test_end=datetime(2022,1,23)
            train = df[:train_end] 
            test = df[train_end + timedelta(days=1):test_end]
            # model=ARMA(train,(1,1,1))
            print(train)
            model = ARIMA(train['Adj Close'], order=(1,1,1))
            results_Arima = model.fit()

            st.write(results_Arima.summary())
            pred_start=test.index[0]
            pred_end=test.index[-1]
            forecast=results_Arima.forecast(12)

            # plt.plot(train,label='Training Data')
            # plt.plot(test,label='Test Data')
            # plt.plot(test.index,forecast[0],label='Predicted Data - AR1MA(1,1,1)')
            # plt.legend(loc='best')
            # plt.grid()
            #fig = px.line(train, y='Adj Close', title='ARIMA Forecast')
            fig = go.Figure()    
            fig.add_trace(go.Scatter(x=train.index, y=train['Adj Close'], mode='lines', name='Train'))
            fig.add_trace(go.Scatter(x=test.index, y=test['Adj Close'], mode='lines', name='Test'))
            #fig.add_trace(go.Scatter(x=test.index, y=forecast[0], mode='lines'))
            st.plotly_chart(fig)   
            

    if choice == 'SARIMAX Forecast':
        st.markdown("<h3 style='text-align: center;'>SARIMAX Forecast</h3>", unsafe_allow_html=True)
        # tickers = ('AMZN','AAPL','TCS.NS','GOOG','ITC.NS','IDEA.NS','BPCL.NS', 'ONGC.NS', 'RDS-A', 'RDS-B','INFY.NS')
        dropdown = st.selectbox('Pick your assets', tickers)

        start = st.date_input('Start',  value=pd.to_datetime('2021-01-01'))
        if len(dropdown):
            df = yf.download(dropdown, start)
            df = df[['Close','Adj Close']]
            df_diff = df.diff(periods=1).dropna().reset_index()
            
            train_end=datetime(2021,12,31)
            test_end=datetime(2022,1,23)
            train = df[:train_end] 
            test = df[train_end + timedelta(days=1):test_end]
            model = SARIMAX(endog =train['Adj Close'],
                                  order=(0,1,1),
                                  seasonal_order=(1,0,1,7),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
            model_Sarima = model.fit()
            st.write(model_Sarima.summary())
            
            pred_start=test.index[0]
            pred_end=test.index[-1]

            #SARIMA_predictions=model_Sarima.predict(test['Adj Close'])

            fig = go.Figure()    
            fig.add_trace(go.Scatter(x=train.index, y=train['Adj Close'], mode='lines', name='Train'))
            fig.add_trace(go.Scatter(x=test.index, y=test['Adj Close'], mode='lines', name='Test'))
            #fig.add_trace(go.Scatter(x=test.index, y=SARIMA_predictions, mode='lines'))
            st.plotly_chart(fig)
            

    if choice == 'Prophet Forecast':
        st.markdown("<h3 style='text-align: center;'>Prophet Forecast</h3>", unsafe_allow_html=True)
        # with st.expander("About the app", expanded=False):
        #     st.write("""
        #             This application allows us to train, evaluate and optimize a time series model in just a few clicks.
        #             """)
        #     st.write("")
        # tickers = ('AMZN','AAPL','TCS.NS','GOOG','ITC.NS','IDEA.NS','BPCL.NS', 'ONGC.NS', 'RDS-A', 'RDS-B','INFY.NS')
        dropdown = st.selectbox('Pick your assets', tickers)
        start_col, end_col = st.columns(2)
        with start_col:
            start = st.date_input('Start',  value=pd.to_datetime('2020-01-01'))
        with end_col:
            end = st.date_input('End',  value=pd.to_datetime('today'))
        if len(dropdown):
            data_tab = st.expander(label='Stock Price Data', expanded=False)
            df = yf.download(dropdown, start, end).reset_index()
            df['Date'] = df['Date'].dt.date
            with data_tab:
                st.dataframe(df.sort_values('Date', ascending=False).set_index('Date'))
        df = df[['Date','Close']].sort_values('Date', ascending=True).rename({'Date':'ds','Close':'y'}, axis='columns')
        n_periods = st.number_input("Number of days to focus on", value=5)
        last_trading_date = df['ds'].tail(1).values[0]
        forecast_from = df['ds'].tail(n_periods).values[0]
        st.write(forecast_from)
        forecast_df = df[df.ds < forecast_from]
        m = Prophet()
        m.fit(forecast_df)
        future = m.make_future_dataframe(periods=n_periods)
        forecast = m.predict(future)
        forecasted_df = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_periods)
        forecasted_df['ds'] = df['ds'].tail(n_periods).values
        forecasted_df['y'] = df['y'].tail(n_periods).values
        st.dataframe(forecasted_df.set_index('ds'))
        st.write("## Overview")
        with st.expander("More info on this plot", expanded=False):
            st.write("""
            This visualization displays several information:
            * The blue line shows the __predictions__ made by the model on both training and validation periods.
            * The black points are the __actual values__ of the target on training period.
            We can use the slider at the bottom or the buttons at the top to focus on a specific time period.
            """)
            st.write("")
        try:    
            st.plotly_chart(plot(m, forecast))
            st.write("## Statistical Test")
        except:
            pass
        
        with st.expander("More info on Statistical Test", expanded=False):
            st.write("""
            The following statistical test can be computed:
            * __Augmented Dickey Fuller (ADF)__ Test: It can be used to test for a unit root in a univariate process in the presence of serial correlation.
            * __Kwiatkowski-Phillips-Schmidt-Shin (KPSS)__ Test: It is a type of Unit root test that tests for the stationarity 
            of a given series around a deterministic trend.
            * __Granger Causality (GC)__ Test:It is for determining whether one time series is useful for forecasting another. 
            """)
        tests = st.selectbox('Pick your Statistical Test',['Augmented Dickey Fuller (ADF)',
                                                           'Kwiatkowski-Phillips-Schmidt-Shin (KPSS)',
                                                           'Granger Causality (GC)'])
        if tests == 'Augmented Dickey Fuller (ADF)':
            series = df.loc[:, 'y'].values
            result = adfuller(series, autolag='AIC')
            st.write('ADF Statistic',round(result[0],3))
            st.write('p-value',round(result[1],3))
            st.write('Critial Values:')
            for key, value in result[4].items():
                st.write(key,round(value,2))

        elif tests == 'Kwiatkowski-Phillips-Schmidt-Shin (KPSS)':
            pass
        else:
            pass

        st.write("## Evaluation Metrics")
        with st.expander("More info on evaluation metrics", expanded=False):
            st.write("""
            The following metrics can be computed to evaluate model performance:
            * __Mean Absolute Percentage Error (MAPE)__: Measures the average absolute size of each error in percentage
            of the truth. This metric is not ideal for low-volume forecasts,
            because being off by a few units can increase the percentage error signficantly.
            It can't be calculated if the true value is 0 (here samples are excluded from calculation if true value is 0).
            * __Symmetric Mean Absolute Percentage Error (SMAPE)__: Slight variation of the MAPE,
            it measures the average absolute size of each error in percentage of the truth summed with the forecast.
            It is therefore a bit more robust to 0 values.
            * __Mean Squared Error (MSE)__: Measures the average squared difference between forecasts and true values.
            This metric is not ideal with noisy data,
            because a very bad forecast can increase the global error signficantly as all errors are squared.
            * __Root Mean Squared Error (RMSE)__: Square root of the MSE.
            This metric is more robust to outliers than the MSE,
            as the square root limits the impact of large errors in the global error.
            * __Mean Absolute Error (MAE)__: Measures the average absolute error.
            This metric can be interpreted as the absolute average distance between the best possible fit and the forecast.
            """)
            st.write("")
            if st.checkbox("Show metric formulas", value=False):
                st.write("If N is the number of distinct dates in the evaluation set:")
                st.latex(r"MAPE = \dfrac{1}{N}\sum_{t=1}^{N}|\dfrac{Truth_t - Forecast_t}{Truth_t}|")
                st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{t=1}^{N}(Truth_t - Forecast_t)^2}")
                st.latex(r"SMAPE = \dfrac{1}{N}\sum_{t=1}^{N}\dfrac{2|Truth_t - Forecast_t]}{|Truth_t| + |Forecast_t|}")
                st.latex(r"MSE = \dfrac{1}{N}\sum_{t=1}^{N}(Truth_t - Forecast_t)^2")
                st.latex(r"MAE = \dfrac{1}{N}\sum_{t=1}^{N}|Truth_t - Forecast_t|")
        metrics = {"MAPE": MAPE, "SMAPE": SMAPE, "MSE": MSE, "RMSE": RMSE, "MAE": MAE}
        metrics_df = forecasted_df[['yhat', 'y']].rename(columns={'yhat':'forecast','y':'truth'})
        colors = ["#002244","#ff0066","#66cccc","#ff9933","#337788","#429e79","#474747","#f7d126","#ee5eab","#b8b8b8"]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.markdown(f"<p style='color: {colors[3]}; " 
                    f"font-weight: bold; font-size: 20px;'> {list(metrics.keys())[0]}</p>", unsafe_allow_html=True,)
        col1.write(round(metrics_df[["truth", "forecast"]].apply(lambda x: metrics[list(metrics.keys())[0]](x.truth, x.forecast), axis=1).sum(),3))
        col2.markdown(f"<p style='color: {colors[3]}; "
                    f"font-weight: bold; font-size: 20px;'> {list(metrics.keys())[1]}</p>", unsafe_allow_html=True,)
        col2.write(round(metrics_df[["truth", "forecast"]].apply(lambda x: metrics[list(metrics.keys())[1]](x.truth, x.forecast), axis=1).sum(),3))
        col3.markdown(f"<p style='color: {colors[3]}; "
                    f"font-weight: bold; font-size: 20px;'> {list(metrics.keys())[2]}</p>", unsafe_allow_html=True,)
        col3.write(round(metrics_df[["truth", "forecast"]].apply(lambda x: metrics[list(metrics.keys())[2]](x.truth, x.forecast), axis=1).sum(),3))
        col4.markdown(f"<p style='color: {colors[3]}; "
                    f"font-weight: bold; font-size: 20px;'> {list(metrics.keys())[3]}</p>", unsafe_allow_html=True,)
        col4.write(round(metrics_df[["truth", "forecast"]].apply(lambda x: metrics[list(metrics.keys())[3]](x.truth, x.forecast), axis=1).sum(),3))
        col5.markdown(f"<p style='color: {colors[3]}; "
                    f"font-weight: bold; font-size: 20px;'> {list(metrics.keys())[4]}</p>", unsafe_allow_html=True,)       
        col5.write(round(metrics_df[["truth", "forecast"]].apply(lambda x: metrics[list(metrics.keys())[4]](x.truth, x.forecast), axis=1).sum(),3))
        st.write("## Error Analysis")
        with st.expander("Forecasting Erros", expanded=False):
            st.write("""
            The below plots help us to to detect patterns in forcasting errors:
            * It shows forecasts vs the ground truth on evaluation period.          
            """)
        fig = px.line(forecasted_df, x="ds", y=["y", "yhat"], hover_data={"variable": True, "value": ":.4f", "ds": False})
        fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(
                        buttons=list([ dict(count=7, label="1w", step="day", stepmode="backward"),
                                        dict(count=1, label="1m", step="month", stepmode="backward"),
                                        dict(count=3, label="3m", step="month", stepmode="backward"),
                                        dict(count=6, label="6m", step="month", stepmode="backward"),
                                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                                        dict(count=1, label="1y", step="year", stepmode="backward"),
                                        dict(step="all")])
            ),
        )
        fig.update_layout(
        yaxis_title='Close Price', legend_title_text="", height=500, width=800, title_text="Forecast vs Truth", title_x=0.5, title_y=1,hovermode="x unified",)
        st.plotly_chart(fig)  
        st.write("## Decomposition")
        with st.expander("More info on this metrics", expanded=False):
            st.write("""
            The forecast generated by time series model is the sum of different contributions:
            * Trend
            * Seasonalities
            * Other factors such as holidays or external regressors
            The following visualization shows this breakdown and allows you to understand how each component contributes
            to the final value forecasted by the model.
            """)
        st.plotly_chart(plot(m, forecast))
        href = f'<a href="">Download Forecast Output</a>'
        st.markdown(href, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
