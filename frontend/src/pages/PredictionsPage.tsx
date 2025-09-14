import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { HTTP_BASE_URL } from "@/config";
import type { ModelConfigResponse } from "@/lib/types/modelConfigResponse";
import { PredictionOutcome } from "@/lib/types/predictionOutcome";
import type { PredictionsResponse } from "@/lib/types/predictionResponse";
import { Bot } from "lucide-react";
import { useEffect, useState, type FC } from "react";

type UpcomingRace = {
  circuitName: string;
  country: string;
  date: Date;
};

const HeroSection = ({
  upcomingRace,
}: {
  upcomingRace: UpcomingRace | null;
}) => {
  const [timeLeft, setTimeLeft] = useState("");
  const seasonOver = false; // make dynamic depending on data
  const nextRace = upcomingRace
    ? {
        // circuitName: "Monza GP",
        // date: new Date("2025-09-07T14:00:00Z"),
        ...upcomingRace,
        flag: "🇮🇹",
        trackImage: "/src/assets/images/monza.png",
      }
    : null;

  useEffect(() => {
    if (seasonOver) return;

    const interval = setInterval(() => {
      const now = new Date().getTime();
      const distance = nextRace.date.getTime() - now;

      if (distance <= 0) {
        setTimeLeft("Race has started!");
        clearInterval(interval);
      } else {
        const days = Math.floor(distance / (1000 * 60 * 60 * 24));
        const hours = Math.floor(
          (distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)
        );
        const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((distance % (1000 * 60)) / 1000);

        setTimeLeft(`${days}d ${hours}h ${minutes}m ${seconds}s`);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <section className="relative">
      <Card className="w-full rounded-2xl shadow-lg overflow-hidden relative">
        {/* Background image with blur & dark overlay */}
        <div
          className="absolute inset-0 bg-cover bg-center blur-xs brightness-100"
          style={{
            backgroundImage:
              "url(https://media.formula1.com/image/upload/t_16by9South/c_lfill,w_3392/q_auto/v1740000000/trackside-images/2023/F1_Grand_Prix_of_Italy/1657349836.webp)",
          }}
        />
        <div className="absolute inset-0 bg-black/40" />

        {/* Foreground content */}
        <CardContent className="relative flex flex-col md:flex-row items-center md:items-start gap-6 p-6">
          {/* Track graphic */}
          <div className="w-full md:w-1/3 flex justify-center">
            <img
              src={nextRace.trackImage}
              alt={nextRace.circuitName}
              className="object-contain h-40"
            />
          </div>

          {/* Text */}
          <div className="flex flex-col justify-center items-center md:items-start text-center md:text-left space-y-2">
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <img
                src="src/assets/images/italy.png"
                alt=""
                width={24}
                height={24}
              />
              {nextRace.circuitName}
            </h2>
            {seasonOver ? (
              <p className="text-lg font-medium text-gray-300">
                The season is over. 🏁
              </p>
            ) : (
              <>
                <p className="text-gray-400">Next race countdown</p>
                <p className="text-xl font-mono font-semibold text-white">
                  {timeLeft}
                </p>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </section>
  );
};

const ModelCard = ({
  modelInfo,
}: {
  modelInfo: ModelConfigResponse | null;
}) => {
  return (
    <Card className="w-full  mt-6 shadow-md rounded-2xl bg-neutral-900 text-gray-200">
      <CardContent className="space-y-4">
        {modelInfo ? (
          <>
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-1">
                Features
              </h3>
              <div className="flex flex-wrap gap-2">
                {modelInfo.features.map((f) => (
                  <span
                    key={f}
                    className="px-2 py-1 bg-neutral-800 rounded-md text-xs text-gray-300"
                  >
                    {f}
                  </span>
                ))}
              </div>
            </div>

            {/* Hyperparameters */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-1">
                Hyperparameters
              </h3>
              <ul className="text-sm space-y-1">
                {Object.entries(modelInfo.params).map(([key, val]) => (
                  <li key={key} className="flex justify-between">
                    <span className="text-gray-400">{key}</span>
                    <span className="text-gray-200 font-medium">
                      {String(val)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Performance */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-1">
                Performance
              </h3>
              <p className="text-sm flex justify-between">
                Accuracy:{" "}
                <span className="font-medium text-white">
                  {Math.round(modelInfo.stats.accuracy * 100)}%
                </span>
              </p>
              <p className="text-sm flex justify-between">
                Precision:{" "}
                <span className="font-medium text-white">
                  {Math.round(modelInfo.stats.precision * 100)}%
                </span>
              </p>
            </div>
          </>
        ) : (
          <>
            <div className="h-75 flex flex-col items-center justify-center">
              <Bot size={75} className="animate-bot-bounce" />
              <span>Model Config...</span>
            </div>
          </>
        )}
        {/* Features */}
      </CardContent>
    </Card>
  );
};

const Progress = ({ value, color }: { value: number; color?: string }) => {
  return (
    <div className="w-full">
      <div className="text-sm font-medium mb-1 text-gray-300">{value}%</div>
      <div className="w-full bg-gray-800 rounded-full h-2">
        <div
          className="h-2 rounded-full transition-all duration-300 shadow-sm"
          style={{
            width: `${value}%`,
            backgroundColor: color || "#fff",
          }}
        />
      </div>
    </div>
  );
};

const PredictionsPage: FC = () => {
  const [outcomeClass, setActiveModel] = useState<PredictionOutcome>(
    PredictionOutcome.ALL
  );
  // const [top3Data, setTop3Data] = useState<any[]>([]);
  // const [winnerData, setWinnerData] = useState<PredictionsResponse[]>([]);
  const [predData, setPredData] = useState<PredictionsResponse[]>([]);
  const [modelConfig, setModelConfig] = useState<ModelConfigResponse | null>(
    null
  );
  const [upcomingRace, setUpcomingRace] = useState<UpcomingRace | null>(null);

  useEffect(() => {
    const fetchPreds = async () => {
      const rsp = await fetch(
        HTTP_BASE_URL + `/predictions?outcome_class=${outcomeClass}`
      );

      if (rsp.ok) {
        const data: PredictionsResponse[] = await rsp.json();
        setPredData(data);
      }
    };

    fetchPreds();
  }, [outcomeClass]);

  useEffect(() => {
    const fetchPreds = async () => {
      setModelConfig(null);
      const rsp = await fetch(
        HTTP_BASE_URL + `/model-config?outcome_class=${outcomeClass}`
      );

      if (rsp.ok) {
        const data: ModelConfigResponse = await rsp.json();
        setModelConfig(data);
      }
    };

    fetchPreds();
  }, [outcomeClass]);

  useEffect(() => {
    const fetchUpcomingRace = async () => {
      const rsp = await fetch(HTTP_BASE_URL + `/upcoming-race`);

      if (rsp.ok) {
        const data = await rsp.json();
        const races: any[] = data["MRData"]["RaceTable"]["Races"];

        let raceInfo = null;
        const curDate = new Date();

        for (const raceData of races) {
          if (new Date(raceData["date"] as string) >= curDate) {
            raceInfo = raceData;
            break;
          }
        }

        if (raceInfo) {
          setUpcomingRace({
            circuitName: raceInfo["Circuit"]["circuitName"],
            country: raceInfo["Circuit"]["Location"]["country"],
            date: new Date(raceInfo["date"] as string),
          });
        }
      }
    };

    fetchUpcomingRace();
  }, []);

  const getTeamLogo = (team: string) =>
    `src/assets/images/${team.toLowerCase()}-logo.png`;

  const getDriverImage = (driver: string) =>
    `src/assets/images/${driver.toLowerCase()}.avif`;

  const getTeamColor = (team: string): string => {
    const colors: Record<string, string> = {
      mercedes: "#27F4D2", // Tiffany Green
      ferrari: "#DC0000", // Red
      red_bull: "#1E41FF", // Blue
      mclaren: "#FF8700", // Orange
      aston_martin: "#006F62", // Green
      alpine: "#00A9E0", // Blue
      haas: "#B8B8B8", // Gray
      williams: "#005F8C", // Blue
      sauber: "#FFB81C", // Yellow
      rb: "#6692FF", // Light Blue
    };

    return colors[team.toLowerCase()] || "#999"; // Default to gray if team not found
  };

  const parseUnderscrore = (value: string) => {
    let parts = value.split("_");
    parts = parts.map((p) => {
      p = p.toLowerCase().trim();
      return p.charAt(0).toUpperCase() + p.slice(1);
    });
    return parts.join(" ");
  };

  const getProbability = (val: PredictionsResponse) => {
    if (outcomeClass === PredictionOutcome.ALL) {
      return val.predicted_probability;
    }

    if (val.predicted_position === "1") {
      return val.predicted_probability;
    }

    return Math.round(100 - val.predicted_probability);
  };

  return (
    <main className="pb-5">
      <div className="max-w-6xl mx-auto">
        <section className="mt-10">
          <HeroSection />
        </section>

        {/* Prediction Table and Model Card*/}
        <section className="mt-10 flex flex-row gap-3">
          <section className="flex-5/8 flex flex-col">
            {/* Snackbar */}
            <div className="w-full h-5 flex flex-row items-center justify-start gap-3">
              {[
                [PredictionOutcome.ALL, "All"],
                [PredictionOutcome.TOP3, "Top 3"],
                [PredictionOutcome.WINNER, "Winners"],
              ].map(([modal, title]) => (
                <Button
                  key={modal}
                  onClick={() => setActiveModel(modal as typeof outcomeClass)}
                  className={`border-b-2 border-b-transparent rounded-none bg-transparent hover:bg-transparent cursor-pointer hover:text-white ${
                    outcomeClass === modal
                      ? "text-white text-shadow-md border-b-red-600"
                      : "text-neutral-500 hover:border-b-1 hover:border-b-gray-800"
                  }`}
                >
                  {title}
                </Button>
              ))}
            </div>

            {/* Tables */}
            <div className="mt-6">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Driver</TableHead>
                    <TableHead>Team</TableHead>
                    <TableHead>Nationality</TableHead>
                    {outcomeClass === PredictionOutcome.ALL && (
                      <TableHead>Predicted Position</TableHead>
                    )}
                    <TableHead>Chance</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody className="text-md font-medium">
                  {predData.length ? (
                    <>
                      {predData.map((val: PredictionsResponse) => (
                        <TableRow key={val.driver.driver_name}>
                          {/* Driver with headshot in colored circle */}
                          <TableCell>
                            <div className="flex items-center gap-3">
                              <div
                                className="w-5 h-5 rounded-full flex items-center justify-center overflow-hidden"
                                style={{
                                  backgroundColor: getTeamColor(
                                    val.driver.constructor
                                  ),
                                }}
                              >
                                <img
                                  src={getDriverImage(val.driver.driver_name)}
                                  alt={val.driver.driver_name}
                                  width={40}
                                  height={40}
                                  className="mt-12"
                                />
                              </div>
                              <span className="font-medium">
                                {parseUnderscrore(val.driver.driver_name)}
                              </span>
                            </div>
                          </TableCell>

                          {/* Team with logo */}
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <div
                                className="w-5 h-5 rounded-full flex items-center justify-center overflow-hidden"
                                style={{
                                  backgroundColor: getTeamColor(
                                    val.driver.constructor
                                  ),
                                }}
                              >
                                <img
                                  src={getTeamLogo(val.driver.constructor)}
                                  alt={val.driver.constructor}
                                  width={12}
                                  height={12}
                                  className=""
                                />
                              </div>
                              <span>
                                {parseUnderscrore(val.driver.constructor)}
                              </span>
                            </div>
                          </TableCell>

                          {/* Nationality */}
                          <TableCell>{val.driver.nationality}</TableCell>

                          {outcomeClass === PredictionOutcome.ALL && (
                            <TableCell>{val.predicted_position}</TableCell>
                          )}

                          {/* Chance with progress bar */}
                          <TableCell>
                            <Progress
                              value={getProbability(val)}
                              color={getTeamColor(val.driver.constructor)}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </>
                  ) : (
                    <TableRow>
                      <TableCell colSpan={4}>
                        <div className="flex items-center justify-center">
                          <span>No predictions</span>
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </section>
          {/* {modelConfig && ( */}
          <section className="flex-3/8">
            <div className="sticky top-2">
              <ModelCard modelInfo={modelConfig} outcomeClass={outcomeClass} />
            </div>
          </section>
          {/* )} */}
        </section>
      </div>
    </main>
  );
};

export default PredictionsPage;
