from pandas import DataFrame
from .utils import (
    PosCat,
    append_drivers_age,
    append_elo_change,
    append_elo_percentile,
    append_elo_rank_in_race,
    append_elo,
    append_field_pos_delta,
    append_dnf_count,
    append_last_n,
    append_avg_position_move,
    append_last_season_wins,
    append_median_race_position,
    append_nationality_encodings,
    append_position_propensity,
    append_sma,
    append_std,
    append_constructor_encodings,
    get_position_category,
)
from ..preprocessing import merge_datasets


def drop_features(df: DataFrame) -> DataFrame:
    return df.drop(
        [
            "circuitId",
            "constructorId",
            "driverId",
            "raceId",
            "position",
            "points",
            "wins",
            "position_numeric",
            "positionText",
            "year",
            "position_driver_standings",
            "points_constructor_standings",
            "position_constructor_standings",
            "dob",
            "positionOrder",
            "constructorRef",
            "driverRef",
            "nationality",
        ],
        axis=1,
    )


def get_dataset(pos_cat: PosCat) -> DataFrame:
    df: DataFrame = merge_datasets()
    # df["target"] = df["positionText"].apply(lambda x: get_position_category(x, pos_cat))
    df["target"] = df["positionOrder"]
    df = append_constructor_encodings(df)
    # df = append_drivers_age(df)
    # df = append_nationality_encodings(df)
    # df["target"] = df["positionText"]

    # df = append_last_n(df, "target", window=2, typ="str")
    # df = append_position_propensity(df, pos_cat)
    # df = append_avg_position_move(df, window=0)
    # df = append_median_race_position(df, window=3)
    # df = append_sma(df, window=0)
    # df = append_std(df, window=0)
    # df = append_elo(df, default_elo=100, k=1)
    # df = append_elo_rank_in_race(df)
    # df = append_elo_percentile(df)
    # df = append_last_n(df, "elo", window=10)
    return df


# df = get_dataset("loose")
# df = df[df["year"] == 2022]
# print(len(df))
if __name__ == "__main__":
    drop_features(get_dataset("loose")).to_csv("file.csv", index=False)
