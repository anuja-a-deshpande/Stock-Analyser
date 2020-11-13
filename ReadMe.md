# Stock Analyser 

```Goal: To predict price variaton to define high frequency trading margins by algorithmically Identifying patterns and corelating stock indices.```

## Approach:
    1. Algorithmic corelation between dependent indices
    - Method: SVR (Epsilon-Support Vector Regression)

    2. Predicting values by identifying patterns in historical data 
    - Method: RMSE (Root mean square error)

## Usage:
    1. StockTestSVR:
       python3 ./<projectdirectory>/StockTestSVR.py
       Input data files : Put below input files in the root of your <projectdirectory>:
            1. stockdataTrain_SVR.csv 
                - Contains two columns : Stock1 Prices and Stock2 prices.
                - We use this data to train our model. And we will try to find out corelation between them.
            2. stockdataTest_SVR.csv 
                - Contains two columns : Stock1 Prices and Stock2 prices.
                - We use this data to test our model. Using Stock1 Prices we will predict  Stock2 prices.
    
    2. StockTestRMSE:
        python3 ./<projectdirectory>/StockTestRMSE.py
        Input data files : Put below input files in the root of your <projectdirectory>:
            1. stockdataTrain_RMSE.csv
                - Contains two columns : Dates and Stock Prices.
                - We use this 65% data to train our model, we will rest of the data as test data. And we will predict future stock prices.


## Future Scope:
    1. Getting End of day(EOD) data using REST API.
    2. Using technical indicators and  oscillators as triggers to run analysis.
    3. Creating mass stock screener.


