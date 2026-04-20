## Dataset Information

**Online Retail Dataset**: https://archive.ics.uci.edu/dataset/502/online+retail+ii

This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.
The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.

### Additional Variable Information

- **InvoiceNo**: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation. 
- **StockCode**: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product. 
- **Description**: Product (item) name. Nominal. 
- **Quantity**: The quantities of each product (item) per transaction. Numeric.	
- **InvoiceDate**: Invoice date and time. Numeric. The day and time when a transaction was generated. 
- **UnitPrice**: Unit price. Numeric. Product price per unit in sterling (Â£). 
- **CustomerID**: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer. 
- **Country**: Country name. Nominal. The name of the country where a customer resides.

The Recency, Frequency, and Monetary (RFM) model is a common method used customer segmentation to analyze customer behavior and predict their engagement. RFM analysis is commomly used in marketing to identify hight-value customers and tailor strategies accordingly.

- **Recency**: Computed as the number of days between the last purchase and the reference date.
- **Frequency**: Number of unique invoices (transactions) per customer.
- **Monetary**: Sum of the total transaction values for each customer.

We defining reference_date as one day after the last transaction, standard practice in Recency-Frequency-Monetary (RFM) analysis for the following reasons:

**1. Ensuring Positive Recency Values**
- Recency measures how many days ago a customer made their last purchase.
- If reference_date = df["InvoiceDate"].max(), then the most recent customers would have Recency = 0 (which can be confusing in analysis).
- By adding one day, the most recent customer will have Recency = 1, making interpretation clearer.

**2. Simulating a Snapshot in Time**
- RFM analysis is typically performed as if at a specific point in time (e.g., "as of today").
- Since transaction data may be historical, setting the reference one day ahead ensures consistency with future analyses.

**3. Avoiding Bias for the Last Purchase Date**
- If we set reference_date = df["InvoiceDate"].max(), the last transaction's recency would be 0, while others have positive values.
- This may introduce bias when segmenting customers, especially in clustering or ranking.