import {
  ArrowTrendingDownIcon,
  ArrowTrendingUpIcon,
  NoSymbolIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import { useEffect, useRef, useState, type FC, type JSX } from "react";

import { BettingService } from "../lib/classes/BettingService";
import { UtilsManager } from "../lib/classes/UtilsManager";
import { BetSide } from "../lib/types/betSide";
import { ChainId } from "../lib/types/networkConfig";
import type { Market } from "../types";

const BettingSlipCard: FC<{
  market: Market;
  setShow: (arg: boolean) => void;
}> = ({ market, setShow }) => {
  const DEFAULT_METAMODAL_MSG = "Please confirm the connection in your wallet";

  const [isAwaitingHealthCheck, setIsAwaitingHealthCheck] =
    useState<boolean>(true);
  const [isHealthy, setIsHealthy] = useState<boolean | undefined>(undefined);
  const [curSide, setSide] = useState<BetSide>(BetSide.BACK);
  const [amount, setAmount] = useState("");

  const [isVisible, setIsVisible] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  const [showMetaMaskModal, setShowMetaMaskModal] = useState<boolean>(false);
  const [metaMaskModalMessage, setMetaMaskModalMessage] = useState<string>(
    DEFAULT_METAMODAL_MSG
  );

  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [bettingService] = useState<BettingService>(
    () => new BettingService(ChainId.SEPOLIA)
  );

  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const rsp = await fetch(
          UtilsManager.BASE_URL +
            `/markets/health?market_id=${market.market_id}`
        );
        if (rsp.ok) {
          setIsHealthy((await rsp.json())["health"]);
        }
      } catch (error) {
      } finally {
        setIsAwaitingHealthCheck(false);
      }
    };

    fetchHealth();
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 50);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const handle = () => document.body.classList.toggle("overflow-hidden");
    handle();

    return () => {
      handle();
    };
  }, []);

  const handleClose = (): void => {
    setIsClosing(true);
    setTimeout(() => {
      setShow(false);
    }, 300);
  };

  const formatOdds = (numerator: number, denominator: number): string => {
    return `${numerator}/${denominator}`;
  };

  const calculatePayout = (stake: number): Record<BetSide, number> => ({
    [BetSide.BACK]: stake * market.back_multiplier,
    [BetSide.LAY]: stake * market.lay_multiplier,
  });

  const numericAmount = parseFloat(amount) || 0;
  const payout = Object.freeze(calculatePayout(numericAmount));

  const sanitizeInput = (value: string) => /^\d*\.?\d*$/.test(value.trim());

  async function handleFormSubmit(e: React.FormEvent): Promise<void> {
    e.preventDefault();
    setError(null);

    if (!sanitizeInput(amount)) {
      setError("Please enter a valid amount.");
      (e.target as HTMLFormElement).reset();
      return;
    }

    try {
      setShowMetaMaskModal(true);
      
      await bettingService.connect();
      await bettingService.placeBet(
        market.market_id,
        new FormData(e.target as HTMLFormElement).get("amount") as string,
        curSide
      );

      setMetaMaskModalMessage("Confirm contract to place bet.");
      setSuccess("Bet placed successfully");
      setTimeout(() => handleClose(), 1000);
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setShowMetaMaskModal(false);
      setMetaMaskModalMessage(DEFAULT_METAMODAL_MSG);
    }
  }

  function renderBettingSlipCardContent(): JSX.Element {
    return (
      <>
        <div className="relative">
          <button
            onClick={handleClose}
            className="absolute right-2 top-2 hover:cursor-pointer"
          >
            <XMarkIcon className="w-7 h-7" />
          </button>

          {/* Market Info */}
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-bold">MV</span>
              </div>
              <div className="w-fit ">
                <h4 className="font-semibold text-gray-900">{market.title}</h4>
                <p className="text-sm text-gray-500">category</p>
              </div>
              <div className="text-right">
                <div className="bg-white px-3 py-1 rounded-lg border border-gray-200">
                  <span className="text-lg font-mono font-bold text-gray-900">
                    {formatOdds(market.numerator, market.denominator)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Bet Type Selection */}
          <div className="p-4">
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Bet Type
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setSide(BetSide.BACK)}
                  className={`p-3 rounded-xl border-2 transition-all duration-200 flex items-center justify-center gap-2 ${
                    curSide === BetSide.BACK
                      ? "border-green-500 bg-green-50 text-green-700"
                      : "border-gray-200 hover:border-gray-300 text-gray-600"
                  }`}
                >
                  <ArrowTrendingUpIcon className="w-5 h-5" />
                  <div className="text-left">
                    <div className="font-semibold">Back</div>
                    <div className="text-xs opacity-75">Bet on win</div>
                  </div>
                </button>

                <button
                  onClick={() => setSide(BetSide.LAY)}
                  className={`p-3 rounded-xl border-2 transition-all duration-200 flex items-center justify-center gap-2 ${
                    curSide === BetSide.LAY
                      ? "border-purple-500 bg-purple-50 text-purple-700"
                      : "border-gray-200 hover:border-gray-300 text-gray-600"
                  }`}
                >
                  <ArrowTrendingDownIcon className="w-5 h-5" />
                  <div className="text-left">
                    <div className="font-semibold">Lay</div>
                    <div className="text-xs opacity-75">Bet against</div>
                  </div>
                </button>
              </div>
            </div>
            <form ref={formRef} onSubmit={handleFormSubmit}>
              {/* Amount Input */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Stake Amount
                </label>
                <div className="relative">
                  <div className="abettingServiceolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <span className="text-gray-500 text-lg">$</span>
                  </div>
                  <input
                    type="number"
                    value={amount}
                    onChange={(e) => {
                      const value = e.target.value;
                      if (!sanitizeInput(value)) {
                        setError("Please enter a valid amount.");
                      } else {
                        setError(null);
                        setAmount(value);
                      }
                    }}
                    name="amount"
                    placeholder="0.00"
                    className="w-full pl-8 pr-4 py-3 border border-gray-300 rounded-xl text-lg font-semibold focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  />
                </div>

                {/* Quick Amount Buttons */}
                <div className="flex gap-2 mt-2">
                  {[10, 25, 50, 100].map((quickAmount) => (
                    <button
                      type="button"
                      key={quickAmount}
                      onClick={() => setAmount(quickAmount.toString())}
                      className="flex-1 py-2 px-3 text-sm border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      ${quickAmount}
                    </button>
                  ))}
                </div>
              </div>

              {/* Calculation Display */}
              {numericAmount > 0 && (
                <div className="mb-4 p-4 bg-gray-50 rounded-xl border border-gray-200">
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Stake:</span>
                      <span className="font-semibold">
                        ${numericAmount.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Potential Profit:</span>
                      <span
                        className={`font-semibold ${
                          curSide === BetSide.BACK
                            ? "text-green-600"
                            : "text-purple-600"
                        }`}
                      >
                        ${payout[curSide].toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between pt-2 border-t border-gray-200">
                      <span className="text-gray-900 font-medium">
                        {curSide === BetSide.BACK
                          ? "Total Return:"
                          : "Max Liability:"}
                      </span>
                      <span className="font-bold text-lg">
                        $ {payout[curSide].toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="px-4 py-3">
                  <p className="text-sm font-normal error text-center">
                    {error}
                  </p>
                </div>
              )}

              {success && (
                <div className="px-4 py-3">
                  <p className="text-sm font-normal success text-center">
                    {success}
                  </p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="space-y-2">
                <button
                  type="submit"
                  disabled={!numericAmount || numericAmount <= 0}
                  className={`w-full py-4 px-6 rounded-xl font-semibold text-lg transition-all duration-200 hover:cursor-pointer ${
                    curSide === BetSide.BACK
                      ? "bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white"
                      : "bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 text-white"
                  }`}
                >
                  {numericAmount > 0 ? (
                    <>
                      Place {curSide === BetSide.BACK ? "Back" : "Lay"} Bet - $
                      {numericAmount.toFixed(2)}
                    </>
                  ) : (
                    "Enter Amount to Continue"
                  )}
                </button>

                <button
                  type="button"
                  onClick={handleClose}
                  className="w-full py-3 px-6 border border-gray-300 rounded-xl font-medium text-gray-700 hover:bg-gray-50 transition-colors hover:cursor-pointer"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>

          {/* Footer */}
          <div className="px-4 py-3 bg-gray-50 border-t border-gray-200 pb-25 md:pb-3">
            <p className="text-xs text-gray-500 text-center">
              Bets are final once placed. Please review your selection
              carefully.
            </p>
          </div>
        </div>
      </>
    );
  }

  function renderAwaitingHealthCheck(): JSX.Element {
    return (
      <div className="w-104 h-168 flex-center flex-col text-center px-5 relative z-10">
        <h3 className="text-lg font-semibold text-gray-900">
          Checking market health
        </h3>
        <div className="flex items-center justify-center space-x-1 mt-4">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
          <div
            className="w-2 h-2 bg-blue-500 rounded-full size-10 animate-bounce"
            style={{ animationDelay: "0.1s" }}
          ></div>
          <div
            className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
            style={{ animationDelay: "0.2s" }}
          ></div>
        </div>
      </div>
    );
  }

  function renderNotHeatlhy(): JSX.Element {
    setTimeout(() => handleClose(), 2000);
    return (
      <div className="w-104 h-168 flex-center flex-col text-center px-5 relative z-10">
        <NoSymbolIcon color="red" className="w-20 h-20" />
        <h3 className="text-lg font-semibold text-gray-900">Market not open</h3>
      </div>
    );
  }

  function renderCardContent(): JSX.Element {
    if (isAwaitingHealthCheck) {
      return renderAwaitingHealthCheck();
    }

    if (isHealthy) {
      return renderBettingSlipCardContent();
    }

    return renderNotHeatlhy();
  }

  return (
    <>
      {showMetaMaskModal && (
        <div className="w-full h-screen fixed top-0 left-0 flex items-center justify-center bg-black/30 backdrop-blur-sm z-101">
          <div className="w-80 h-80 flex items-center justify-center flex-col relative border border-gray-200 rounded-3xl bg-white shadow-2xl p-8">
            <div className="relative z-10 mb-6">
              <img
                src="/src/assets/images/MetaMask_Fox.png"
                alt="MetaMask Fox"
                className="w-16 h-16 object-contain"
              />
            </div>

            <div className="text-center space-y-2 relative z-10">
              <h3 className="text-lg font-semibold text-gray-900">
                {metaMaskModalMessage}
              </h3>
              <p className="text-sm text-gray-500"></p>
              <div className="flex items-center justify-center space-x-1 mt-4">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: "0.1s" }}
                ></div>
                <div
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: "0.2s" }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="w-screen h-screen fixed top-0 left-0 z-100 backdrop-blur-xs">
        {/* Backdrop for mobile */}
        <div
          className={`fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden transition-opacity duration-300 ${
            isVisible && !isClosing
              ? "opacity-100"
              : "opacity-0 pointer-events-none"
          }`}
          onClick={handleClose}
        />

        {/* Desktop: Slide down from top */}
        <div
          className={`hidden md:block fixed top-0 left-1/2 transform -translate-x-1/2 z-50 transition-all duration-500 ease-out  ${
            isVisible && !isClosing
              ? "translate-y-8 opacity-100"
              : "-translate-y-full opacity-0"
          }`}
        >
          <div className="max-w-md max-h-176 rounded-2xl shadow-2xl border border-gray-200 bg-white overflow-hidden overflow-y-auto">
            {renderCardContent()}
          </div>
        </div>

        {/* Mobile: Slide up from bottom */}
        <div
          className={`md:hidden fixed bottom-0 left-0 right-0 z-50 transition-all duration-500 ease-out ${
            isVisible && !isClosing
              ? "translate-y-0 opacity-100"
              : "translate-y-full opacity-0"
          }`}
        >
          <div className="bg-white rounded-t-3xl shadow-2xl border-t border-gray-200 max-h-[90vh] overflow-y-auto">
            {renderCardContent()}
          </div>
        </div>
      </div>
    </>
  );
};

export default BettingSlipCard;
