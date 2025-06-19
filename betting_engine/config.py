from json import dumps
from multiprocessing import Queue
from web3 import AsyncWeb3
from web3.contract import AsyncContract
from config import BE_CONTRACT_ADDR, INFURA_API_KEY, USDT_CONTRACT_ADDR

PROVIDER = AsyncWeb3(
    AsyncWeb3.AsyncHTTPProvider(f"https://sepolia.infura.io/v3/{INFURA_API_KEY}")
)

BETTING_ESCROW_CONTRACT: AsyncContract = PROVIDER.eth.contract(
    address=BE_CONTRACT_ADDR,
    abi=dumps(
        [
            {
                "inputs": [
                    {"name": "marketId", "type": "uint256"},
                    {"name": "winner", "type": "address"},
                    {"name": "multiplier", "type": "uint16"},
                ],
                "name": "withdraw",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function",
            },
            {
                "inputs": [
                    {"indexed": True, "name": "participant", "type": "address"},
                    {"indexed": True, "name": "marketId", "type": "uint256"},
                    {"indexed": False, "name": "amount", "type": "uint256"},
                    {"indexed": False, "name": "side", "type": "uint8"},
                ],
                "name": "BetPlaced",
                "type": "event",
                "anonymous": False,
            },
        ]
    ),
)

USDT_CONTRACT: AsyncContract = PROVIDER.eth.contract(
    address=USDT_CONTRACT_ADDR,
    abi=dumps(
        [
            {
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "stateMutability": "view",
                "type": "function",
            },
        ]
    ),
)


MATCHING_ENGINE_QUEUE: Queue
