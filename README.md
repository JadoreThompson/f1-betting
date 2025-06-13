# **Overview**

---

This project is a Formula 1 betting application that integrates live market data, blockchain payments, and predictive analytics. The backend is built with FastAPI, offering a fast and scalable foundation, while transactions are handled through a custom smart contract on the blockchain to ensure security and transparency. A proprietary prediction model powers the betting insights, supported by a dedicated data pipeline. The frontend is developed using React, delivering a responsive and intuitive user experience.

# **Pre-requisites**

A working knowledge of Random Forests ****and their hyperparameters is helpful, particularly when exploring the machine learning components of the system. However, the web application encompasses much more than just the model—it includes a data pipeline, server infrastructure, and various frontend/backend integrations.

To fully understand how each part interacts, and to confidently contribute to or deploy the system, it's important to read through the rest of the documentation. Other sections provide both a holistic overview of the architecture and detailed explanations of each component’s role, configuration, and dependencies.

# **Requirements**

- Python 3.12
- Redis
- Postgres (or dialect compatible db)

# **Installation**

```bash
git clone https://github.com/JadoreThompson/f1-betting

pip install -r requirements.txt # Install dependencies

python __main__.py # Start the whole application
```

# Documentation

## Betting Engine Module

The betting_engine module is the heart of the F1 betting application's backend. It is a self-contained, asynchronous system responsible for managing the entire lifecycle of a bet, from market creation and order matching to data polling and on-chain settlement. It operates as a set of interconnected components that run in parallel to provide a real-time, robust betting experience.

The module is primarily composed of the following key components:

- **Pollers**: Asynchronous workers that fetch F1 race data and settlement information.
- **Market Pipeline**: A machine learning pipeline that uses the polled data to generate predictive odds and create new betting markets.
- **Matching Engine**: The core logic that processes user bets, matches opposing sides (BACK/LAY), and manages order books.
- **Pusher**: A batching and notification service that efficiently updates the database and broadcasts state changes via Redis.
- **Order and OrderBook**: Data structures representing individual bets and market-specific collections of bets.

**Pollers (`betting_engine/pollers/`)**

The polling system continuously fetches data from external APIs to keep the application's database up-to-date and to trigger key events like market settlement.

This is the primary data-gathering component. It runs in a continuous loop to perform the following tasks:

- Fetches the F1 race schedule for the current season.
- Systematically polls for results from past events (Qualifying, Sprints, Grand Prix).
- Parses the raw JSON data from the API into structured Python dataclasses (QualiResult, GrandPrixResult, etc.).
- Persists this structured data into the PostgreSQL database, creating or updating records for drivers, constructors, circuits, and race results.
- Calculates and persists driver and constructor standings after each race.
- After successfully polling new data, it triggers the MarketPipeline to generate new betting markets for upcoming races.

This poller is responsible for triggering the settlement of markets. Its workflow is as follows:

1. Identifies the next upcoming Grand Prix and calculates the time until the race starts.
2. Sleeps until the race is scheduled to begin.
3. Once the race starts, it closes all open markets for that race round, preventing any new bets.
4. It then enters a polling loop, repeatedly checking the API for the final race results.
5. When the results are published, it sends the outcome data to the MatchingEngine to initiate the settlement process for all bets on that market.

**Market Pipeline (`betting_engine/market_pipeline.py`)**

The MarketPipeline is the predictive analytics engine of the application. It leverages machine learning models to create new betting markets with calculated odds.

**Workflow:**

1. **Load Data**: It queries the database to load all relevant historical data (race results, standings, qualifying, etc.) into Pandas DataFrames.
2. **Feature Engineering**: It processes the raw data, creating advanced features necessary for the models, such as driver ELO ratings, recent performance metrics, and podium finishes.
3. **Prediction**: It loads pre-trained Random Forest models (winner_v1, top3_v1) and uses them to predict the probability of each driver winning the race or finishing in the top 3.
4. **Market Creation**: It converts these probabilities into fractional odds (e.g., 8/1).
5. **Persistence**: It inserts these newly generated markets (e.g., "Max Verstappen to win," "Lando Norris to finish Top 3") into the markets table in the database, making them available for users to bet on.

**Matching Engine (`betting_engine/matching_engine.py`)**

The MatchingEngine is the central processing unit for all betting activity. It runs continuously, consuming events from a multiprocessing queue and taking action accordingly.

- **Order Processing**: When a user places a bet (Topic.CREATE), the engine creates an Order object.
- **Order Matching**: It immediately attempts to match the new order against existing orders in the corresponding OrderBook. It matches BACK (for an outcome) and LAY (against an outcome) bets of the same amount. The matching logic ensures that orders are filled completely to avoid fragmented expected value.
- **Order Book Management**: If an order cannot be matched immediately, it is added to the OrderBook to await a matching order.
- **Market Settlement**: When it receives a settlement instruction from the SettlementPoller (Topic.SETTLE), it directs the relevant OrderBook to settle all its bets based on the winning side.

**Order and OrderBook (`betting_engine/order.py, betting_engine/orderbook.py`)**

These classes provide the core data structures for managing bets.

Represents a single, atomic bet placed by a user. It contains the bet's details (payload), its side (BACK/LAY), and its current status (e.g., PENDING, FILLED). It also tracks the remaining amount to be filled.

Manages all Order objects for a single market (e.g., "Charles Leclerc to win").

- It maintains two collections: bids for BACK orders and asks for LAY orders.
- It provides methods to add and remove orders as they are placed and filled.
- Its most critical function is settle(), which handles the entire payout process. When called by the MatchingEngine, it determines the winning and losing orders, calculates the payout for each winner, and triggers the on-chain withdrawal transaction.

**On-Chain Settlement Integration:**

The _handle_payout method within OrderBook is where the application interacts with the blockchain. For each winning bet, it:

1. Builds a transaction to call the withdraw function on the custom BE_CONTRACT (Betting Engine Smart Contract).
2. Signs the transaction using the server's private key.
3. Sends the raw transaction to the Ethereum network (Sepolia testnet).
4. Upon successful broadcast, it records the transaction hash, payout amount, and updates the bet's status to SETTLED in the database.

**Pusher (`betting_engine/pusher.py`)**

The Pusher is an efficiency and real-time notification component that decouples database writes and pub/sub notifications from the main engine logic.

- **Batching**: Instead of writing to the database for every small change, the MatchingEngine and OrderBook append update payloads to the Pusher's queues. The Pusher then processes these updates in batches, reducing database load.
- **Dual Queues**: It uses a "fast" queue for high-priority updates (e.g., order fills) and a "slow" queue for less urgent ones, ensuring a responsive user experience where needed.
- **Publishing**: After committing a batch of updates to the database, it publishes the changes to a Redis pub/sub channel (ORDER_UPDATE_CHANNEL). Frontend clients can subscribe to this channel to receive real-time updates on their bet statuses without needing to poll the server.
- **Concurrency Safety**: It uses a LockClient to ensure that database write operations are atomic and safe from race conditions.

## Models

### Categories

```python
class LoosePositionCategory(str, Enum):
    DNF = "0"        # Did Not Finish
    TOP_3 = "1"      # Finished in positions 1–3
    TOP_5 = "2"      # Finished in positions 4–5
    TOP_10 = "3"     # Finished in positions 5–10
    TOP_20 = "4"     # Finished in positions 10–20 (i.e., completed the race)

class TightPositionCategory(str, Enum):
    DNF = "0"        # Did Not Finish
    FIRST = "1"      # Finished 1st
    SECOND = "2"     # Finished 2nd
    THIRD = "3"      # Finished 3rd

class Top3PositionCategory(str, Enum):
    TOP3 = "1"       # Finished in positions 1–3
    NOT_TOP_3 = "0"  # Finished outside top 3 or DNF

class WinnerPositionCategory(str, Enum):
    WINNER = "1"     # Finished 1st
    NOT_WINNER = "0" # Did not win (includes 2nd–20th and DNF)

```

### winner_v1

Implemented through a Random Forest classifier. This model was trained on data from 2017 to 2022, tested on 2023 and evaluated on 2024. This model takes the following set of features to predict whether a driver will win or not:

Input Features

- `grid` *(int)* – Starting grid position.
- `position_quali` *(int)* – Final position achieved during qualifying.
- `prev_wins` *(float)* – Total number of wins the driver had before the current race within the season.
- `prev_points` *(float)* – Total accumulated points by the driver before the race within the season.
- `prev_position_constructor_standings` *(float)* – Constructor's last standing position prior to this race within the season.
- `elo` *(float)* – Driver’s ELO rating at the start of the race.
- `elo_change` *(float)* – Change in ELO rating from the previous race.
- `last_target_1` to `last_target_6` *(object)* – The driver's last 6 race results, encoded categorically within the season.
- `last_podiums_0` *(float)* – Number of podium finishes within the season so far.

The nuance of this model is that it’s trained to classify each driver on the loose category position with the inference being performed with the winner category. Revealing these metrics:

| **Precision** | **Score** |
| --- | --- |
| Precision | 88% |
| Recall | 58% |

### top3_v1

Implemented through a Random Forest classifier. This model was trained on data from 2017 to 2022, tested on 2023 and evaluated on 2024. This model takes the following set of features to predict whether a driver will finish in the top3 or not:

Input Features

- `grid` *(int)* – Starting grid position.
- `position_quali` *(int)* – Final position achieved during qualifying.
- `prev_wins` *(float)* – Total number of wins the driver had before the current race within the season.
- `prev_points` *(float)* – Total accumulated points by the driver before the race within the season.
- `prev_position_constructor_standings` *(float)* – Constructor's last standing position prior to this race within the season.
- `elo` *(float)* – Driver’s ELO rating at the start of the race.
- `elo_change` *(float)* – Change in ELO rating from the previous race.
- `last_target_1` to `last_target_6` *(object)* – The driver's last 6 race results, encoded categorically within the season.
- `last_podiums_0` *(float)* – Number of podium finishes within the season so far.
- `prev_season_wins` *(float)* – Total number of wins the driver achieved in the previous season.

Much like the winner_v1 model, this model was trained to classify each driver in the loose category. With inference being done using the top3 category.