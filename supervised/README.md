 Please help me write a supervised trading model, based on the papers and ensure to follow this instructions. The intetntion is to create a model that learns different points where it should trade and how many. Then calculate this places where should learn to trade and how many. Then the model should learn to recognize the patterns to execute trades over new data.

 First write a general description of the code and then the code inside a block of code. Please keep all the code inside a  single file. 

 Please use plotply to make plots. The plots must show stock value, portfolio value, credit value over time. The plots must show buy trades and sell-loose and sell-win. Please add all the necessary metrics in the plots, to see what is happening at each or how did the bot did.
Please, a single tab for all the plots!

 Please use tqdm as much as possible, to see what is happening at each step. Please print the evalulation metrics in the terminal, with the results of the model. Please use something like forward moving. 

Please use this data. It contains about 130k samples of BTC-USD, one minute intervals. Separete the training and test dataset:

```
def load_data() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/H3cth0r/stonks-data/refs/heads/main/data/CRYPTO/BTC-USD/data_0.csv"
    column_names = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    df = pd.read_csv(
        url, skiprows=[1, 2], header=0, names=column_names,
        parse_dates=["Datetime"], index_col="Datetime",
        dtype={"Volume": "int64"}, na_values=["NA", "N/A", ""],
        keep_default_na=True,
    )
    df = df.sort_index()
    return df
```

If you need to calculate any indicators, please use python module `ta`.

Please use mps if possible na d please make sure to follow the procedure and configuration from the paper.

Also write the pip install command to install the required dependencies.

Please make sure to have a separated figure per plot or line, beacause portfolio value, stock value and credit can have different scales. Also plot the holdings. The plots and figures must be in the same window, but in separated figures. 

Make sure to correctly identify if it was win or loose sell. 

Please make sure to use correctly the data and divide it into train and test.

Make sure to use correctly the best reward function described in the papers.

Please make sure this does not overfits. Then apply some walk-forward method.

Maybe add the lstm model. Also plot the predictions in the plots. Maybe please train the lstm, print and plot the results of this lstm training and then start using this trained model for the reinforcement learning. Make sure the lstm is well trained.

Maybe use smaller batch sizes, like 20_000 steps max? so that it doesnt take that long and it doesnt overfit and that it trains better. The intention is for intraday, then please consider batches or episode intraday size. This should also speed up training and will align to day trading behvaiour.

Please, if you are going to train an lstm, make sure to correctly train it for as long as it needs, so that it correctly works later with another model.

Please test the model with new data, the idea is that it learns to automatically trade.
Make sure to train the model/models as much as possible! make sure you are training the models correctly.
