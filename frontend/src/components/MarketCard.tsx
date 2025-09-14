import { ArrowTrendingUpIcon } from "@heroicons/react/24/outline";
import { type FC } from "react";
import { UtilsManager } from "../lib/classes/UtilsManager";
import type { Market } from "../types";

const MarketCard: FC<{
  market: Market;
  category: string;
  setShowBetSlip: (arg: boolean) => void;
}> = ({ market, category, setShowBetSlip }) => {
  const formatOdds = (numerator: number, denominator: number): string => {
    return `${numerator}/${denominator}`;
  };

  const calculateImpliedProbability = (
    numerator: number,
    denominator: number
  ): string => {
    const decimal = numerator / denominator + 1;
    const probability = (1 / decimal) * 100;
    return probability.toFixed(0);
  };

  const probability = calculateImpliedProbability(
    market.numerator,
    market.denominator
  );

  return (
    <>
      <div
        className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-lg transition-all duration-200 hover:border-blue-200 cursor-pointer group"
        onClick={() => setShowBetSlip(true)}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <span className="text-white text-xs font-bold">
                {market.title
                  .split(" ")
                  .map((n) => n[0])
                  .join("")
                  .slice(0, 2)}
              </span>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
                {UtilsManager.toCamelCase(market.title)}
              </h4>
              <span className="text-xs text-gray-500">{category}</span>
            </div>
          </div>
          <div className="text-right">
            <div className="bg-gray-50 px-2 py-1 rounded-lg border">
              <span className="text-sm font-mono font-bold text-gray-900">
                {formatOdds(market.numerator, market.denominator)}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="text-xs text-gray-500">Implied probability</div>
            <div className="text-sm font-semibold text-green-600">
              {probability}%
            </div>
          </div>
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <ArrowTrendingUpIcon className="w-3 h-3" />
            <span>$124k vol.</span>
          </div>
        </div>
      </div>
    </>
  );
};

export default MarketCard;
