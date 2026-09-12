import unittest

from player_props.injuries import adjust_prediction_for_injury, find_player_injury, get_injury_report


class InjuryMigrationTests(unittest.TestCase):
    def test_get_injury_report_uses_real_nflreadpy_status_values(self):
        injuries_df = get_injury_report(use_cache=False)
        self.assertFalse(injuries_df.empty, "Injury report should not be empty")

        status_rows = injuries_df[
            injuries_df["status"].fillna("").astype(str).str.strip().ne("")
        ]
        self.assertGreater(len(status_rows), 0, "report_status should be populated for at least some players")

        info = find_player_injury("Aaron Donald", injuries_df)
        self.assertIsNotNone(info, "Aaron Donald should be found in the current injury dataset")
        self.assertTrue(
            str(info.get("status", "")).lower() in {"out", "questionable", "doubtful", "probable"},
            "Matched injury status should be a real game designation",
        )

    def test_adjust_prediction_for_injury_reduces_outlier_players(self):
        injuries_df = get_injury_report(use_cache=False)
        info = find_player_injury("Aaron Donald", injuries_df)
        self.assertIsNotNone(info)

        prediction = {
            "prob_over": 0.66,
            "prob_under": 0.34,
            "confidence": 0.66,
            "injury_note": "",
        }
        adjusted = adjust_prediction_for_injury(prediction.copy(), info)
        self.assertTrue(
            adjusted is None or adjusted.get("confidence", 0.0) < 0.66,
            "Injury adjustment should reduce confidence for a notable injury",
        )


if __name__ == "__main__":
    unittest.main()
