// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

// TODO: Improve robustness
contract BettingEscrow is Ownable, ReentrancyGuard {
    IERC20 public usdtToken;
    mapping(uint256 => address[]) public marketParticipants;
    mapping(uint256 => uint256) public marketEscrow;

    event BetPlaced(address indexed participant, uint256 indexed marketId, uint256 amount);
    event ParticipantPaidOut(address indexed participant, uint256 indexed marketId, uint256 amount);

    constructor(address _usdtTokenAddress) Ownable(msg.sender) {
        require(_usdtTokenAddress != address(0), "Invalid token address");
        usdtToken = IERC20(_usdtTokenAddress);
    }

    function containsParticipant(
        uint256 marketId,
        address participant
    ) public view returns (bool) {
        address[] storage participants = marketParticipants[marketId];
        for (uint256 i = 0; i < participants.length; i++) {
            if (participants[i] == participant) {
                return true;
            }
        }
        return false;
    }

    function placeBet(
        uint256 marketId,
        uint256 amount
    ) external nonReentrant {
        require(msg.sender != address(0), "Invalid sender address");
        require(!containsParticipant(marketId, msg.sender), "Already participated in this market");
        require(amount > 0, "Bet amount must be greater than zero");
        require(usdtToken.transferFrom(msg.sender, address(this), amount), "Transfer failed");

        marketEscrow[marketId] += amount;
        marketParticipants[marketId].push(msg.sender);
        emit BetPlaced(msg.sender, marketId, amount);
    }

    function removeParticipant(uint256 marketId, address participant) internal {
        address[] storage participants = marketParticipants[marketId];
        for (uint256 i = 0; i < participants.length; i++) {
            if (participants[i] == participant) {
                participants[i] = participants[participants.length - 1];
                participants.pop();
                break;
            }
        }
    }

    function withdraw(
        uint256 marketId,
        address winner,
        uint256 amount
    ) external onlyOwner nonReentrant {
        require(winner != address(0), "Invalid winner address");
        require(containsParticipant(marketId, winner), "Winner did not participate in this market");
        require(marketEscrow[marketId] >= amount, "Insufficient escrowed funds");
        require(usdtToken.transfer(winner, amount), "Transfer failed");

        marketEscrow[marketId] -= amount;
        removeParticipant(marketId, winner);
        emit ParticipantPaidOut(winner, marketId, amount);
    }
}