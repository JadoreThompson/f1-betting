// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

// TODO: Improve robustness
contract BettingEscrow is Ownable, ReentrancyGuard {
    IERC20 public usdtToken;
    mapping(uint256 => mapping(address => uint256)) public marketParticipantAmounts; // marketId => address => amount
    mapping(uint256 => mapping(address => bool)) public marketParticipantExists;
    mapping(uint256 => uint256) public marketEscrow;
    uint256 public volume;
    uint256 public activeBetCount;

    event BetPlaced(address indexed participant, uint256 indexed marketId, uint256 amount, uint8 side);
    event ParticipantPaidOut(address indexed participant, uint256 indexed marketId, uint256 amount);

    constructor(address _usdtTokenAddress) Ownable(msg.sender) {
        require(_usdtTokenAddress != address(0), "Invalid token address");
        usdtToken = IERC20(_usdtTokenAddress);
    }

    function containsParticipant(
        uint256 marketId,
        address participant
    ) public view returns (bool) {
        return marketParticipantExists[marketId][participant];
    }

    function placeBet(
        uint256 marketId,
        uint256 amount,
        uint8 side
    ) external nonReentrant {
        require(side == 0 || side == 1, "Invalid side");
        require(msg.sender != address(0), "Invalid sender address");
        require(amount > 0, "Bet amount must be greater than zero");
        require(usdtToken.transferFrom(msg.sender, address(this), amount), "Transfer failed");

        activeBetCount += 1;
        volume += amount;
        
        marketEscrow[marketId] += amount;
        marketParticipantExists[marketId][msg.sender] = true;
        marketParticipantAmounts[marketId][msg.sender] += amount;

        emit BetPlaced(msg.sender, marketId, amount, side);
    }


    function removeParticipant(uint256 marketId, address participant) internal {
        delete marketParticipantExists[marketId][participant];
        delete marketParticipantAmounts[marketId][participant];
    }

    function withdraw(
        uint256 marketId,
        address winner,
        uint16 mutliplier
    ) external onlyOwner nonReentrant {
        require(winner != address(0), "Invalid winner address");
        require(containsParticipant(marketId, winner), "Winner did not participate in this market");
        
        uint256 amount = marketParticipantAmounts[marketId][winner] * mutliplier;
        require(marketEscrow[marketId] >= amount, "Insufficient escrowed funds");
        require(usdtToken.transfer(winner, amount), "Transfer failed");

        activeBetCount -= 1;
        marketEscrow[marketId] -= amount;
        removeParticipant(marketId, winner);
        emit ParticipantPaidOut(winner, marketId, amount);
    }
}
