const { ethers } = require("hardhat");

const WALLET = new ethers.Wallet(process.env.PRIVATE_KEY, ethers.provider);

async function deployUSDTToken() {
  const ContractFactory = await ethers.getContractFactory("USDTToken");
  const contract = await ContractFactory.deploy();

  await contract.waitForDeployment();

  console.log("Contract deployed to:", await contract.getAddress());

  const walletFromContract = contract.connect(WALLET);
  await walletFromContract.transfer(
    "0xec74c989ba1dd95f0b63e8d47d421e678d5eb7b5",
    5000000
  );
}

async function deployBettingEscrow() {
  const usdtAddress = "0x92A1c620751ba38e885461c3e356D41a226962f3";

  const bettingEscrowContract = await (
    await ethers.getContractFactory("BettingEscrow")
  ).deploy(usdtAddress);
  const bettingEscrowContractAddress = await bettingEscrowContract.getAddress();

  await bettingEscrowContract.waitForDeployment();
  console.log("Contract deployed to:", bettingEscrowContractAddress);

  // Get the USDT token contract instance
  const ERC20_ABI = [
    "function approve(address spender, uint256 amount) external returns (bool)",
    "function allowance(address owner, address spender) external view returns (uint256)",
    "function balanceOf(address account) external view returns (uint256)",
  ];

  const usdtContract = new ethers.Contract(usdtAddress, ERC20_ABI, WALLET);
  const betAmount = 3_000_000;

  // Check if we have enough balance
  const balance = await usdtContract.balanceOf(WALLET.address);
  console.log("USDT balance:", ethers.formatUnits(balance, 6));

  if (balance < betAmount) {
    throw new Error("Insufficient USDT balance");
  }

  // Check current allowance
  const currentAllowance = await usdtContract.allowance(
    WALLET.address,
    bettingEscrowContractAddress
  );
  console.log("Current allowance:", ethers.formatUnits(currentAllowance, 6));

  // Approve the contract to spend tokens if needed
  if (currentAllowance < betAmount) {
    console.log("Approving contract to spend USDT...");
    const approveTx = await usdtContract.approve(
      bettingEscrowContractAddress,
      betAmount
    );
    await approveTx.wait();
    console.log("Approval confirmed!");

    // Verify approval
    const newAllowance = await usdtContract.allowance(
      WALLET.address,
      bettingEscrowContractAddress
    );
    console.log("New allowance:", ethers.formatUnits(newAllowance, 6));
  }

  // Now place the bet
  const walletFromContract = bettingEscrowContract.connect(WALLET);
  console.log("Placing bet...");
  const betTx = await walletFromContract.placeBet(1, betAmount);
  await betTx.wait();
  console.log("Bet placed successfully!");
}

deployBettingEscrow().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
