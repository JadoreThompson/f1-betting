from json import dumps
from web3 import AsyncWeb3
from web3.contract import AsyncContract
from config import BE_CONTRACT_ADDR, INFURA_API_KEY, USDT_CONTRACT_ADDR

PROVIDER = AsyncWeb3(
    AsyncWeb3.AsyncHTTPProvider(f"https://sepolia.infura.io/v3/{INFURA_API_KEY}")
)

BE_CONTRACT: AsyncContract = PROVIDER.eth.contract(
    address=BE_CONTRACT_ADDR,
    abi=dumps(
        [
            {
                "inputs": [
                    {"name": "marketId", "type": "uint256"},
                    {"name": "winner", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                ],
                "name": "withdraw",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function",
            }
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
