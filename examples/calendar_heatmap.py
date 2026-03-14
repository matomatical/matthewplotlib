"""
Calendar heatmap of daily maximum temperatures in Oxford, 2025.

Temperature data from Open-Meteo Archive API (51.75°N, 1.25°W).
"""

import tyro
import matthewplotlib as mp
import calendar
import datetime


# Daily maximum temperature (°C), Oxford, 2025
# Source: Open-Meteo Archive API, https://open-meteo.com/
DATA = {
    datetime.date(2025,  1,  1): 11.2, datetime.date(2025,  1,  2):  3.0,
    datetime.date(2025,  1,  3):  4.0, datetime.date(2025,  1,  4):  2.3,
    datetime.date(2025,  1,  5): 11.6, datetime.date(2025,  1,  6): 11.2,
    datetime.date(2025,  1,  7):  4.1, datetime.date(2025,  1,  8):  1.7,
    datetime.date(2025,  1,  9):  2.8, datetime.date(2025,  1, 10):  1.0,
    datetime.date(2025,  1, 11):  1.3, datetime.date(2025,  1, 12):  4.5,
    datetime.date(2025,  1, 13):  5.3, datetime.date(2025,  1, 14):  9.1,
    datetime.date(2025,  1, 15):  8.1, datetime.date(2025,  1, 16):  7.1,
    datetime.date(2025,  1, 17):  6.4, datetime.date(2025,  1, 18):  3.2,
    datetime.date(2025,  1, 19):  2.7, datetime.date(2025,  1, 20):  4.8,
    datetime.date(2025,  1, 21):  5.2, datetime.date(2025,  1, 22):  5.1,
    datetime.date(2025,  1, 23):  7.2, datetime.date(2025,  1, 24): 10.4,
    datetime.date(2025,  1, 25):  6.7, datetime.date(2025,  1, 26):  7.3,
    datetime.date(2025,  1, 27):  7.9, datetime.date(2025,  1, 28):  9.2,
    datetime.date(2025,  1, 29):  7.7, datetime.date(2025,  1, 30):  7.2,
    datetime.date(2025,  1, 31):  8.8,
    datetime.date(2025,  2,  1):  7.1, datetime.date(2025,  2,  2):  7.6,
    datetime.date(2025,  2,  3):  9.1, datetime.date(2025,  2,  4): 10.8,
    datetime.date(2025,  2,  5):  8.9, datetime.date(2025,  2,  6):  5.9,
    datetime.date(2025,  2,  7):  3.8, datetime.date(2025,  2,  8):  4.4,
    datetime.date(2025,  2,  9):  7.1, datetime.date(2025,  2, 10):  4.1,
    datetime.date(2025,  2, 11):  4.8, datetime.date(2025,  2, 12):  5.7,
    datetime.date(2025,  2, 13):  4.0, datetime.date(2025,  2, 14):  4.6,
    datetime.date(2025,  2, 15):  5.3, datetime.date(2025,  2, 16):  4.4,
    datetime.date(2025,  2, 17):  5.2, datetime.date(2025,  2, 18):  6.6,
    datetime.date(2025,  2, 19):  8.0, datetime.date(2025,  2, 20): 13.3,
    datetime.date(2025,  2, 21): 14.2, datetime.date(2025,  2, 22): 12.4,
    datetime.date(2025,  2, 23): 10.4, datetime.date(2025,  2, 24): 11.9,
    datetime.date(2025,  2, 25): 10.2, datetime.date(2025,  2, 26):  8.8,
    datetime.date(2025,  2, 27):  9.4, datetime.date(2025,  2, 28):  8.9,
    datetime.date(2025,  3,  1):  8.9, datetime.date(2025,  3,  2):  8.9,
    datetime.date(2025,  3,  3): 10.4, datetime.date(2025,  3,  4): 11.1,
    datetime.date(2025,  3,  5): 12.8, datetime.date(2025,  3,  6): 15.3,
    datetime.date(2025,  3,  7): 14.9, datetime.date(2025,  3,  8): 16.0,
    datetime.date(2025,  3,  9): 16.9, datetime.date(2025,  3, 10):  9.8,
    datetime.date(2025,  3, 11):  7.4, datetime.date(2025,  3, 12):  6.8,
    datetime.date(2025,  3, 13):  7.2, datetime.date(2025,  3, 14):  7.8,
    datetime.date(2025,  3, 15):  7.8, datetime.date(2025,  3, 16):  7.4,
    datetime.date(2025,  3, 17):  7.8, datetime.date(2025,  3, 18): 10.8,
    datetime.date(2025,  3, 19): 13.9, datetime.date(2025,  3, 20): 18.0,
    datetime.date(2025,  3, 21): 16.0, datetime.date(2025,  3, 22): 15.4,
    datetime.date(2025,  3, 23): 13.4, datetime.date(2025,  3, 24): 12.1,
    datetime.date(2025,  3, 25): 14.4, datetime.date(2025,  3, 26): 15.1,
    datetime.date(2025,  3, 27): 13.4, datetime.date(2025,  3, 28): 12.2,
    datetime.date(2025,  3, 29): 11.8, datetime.date(2025,  3, 30): 14.4,
    datetime.date(2025,  3, 31): 15.4,
    datetime.date(2025,  4,  1): 13.4, datetime.date(2025,  4,  2): 14.9,
    datetime.date(2025,  4,  3): 17.0, datetime.date(2025,  4,  4): 19.5,
    datetime.date(2025,  4,  5): 15.6, datetime.date(2025,  4,  6): 14.2,
    datetime.date(2025,  4,  7): 13.6, datetime.date(2025,  4,  8): 13.6,
    datetime.date(2025,  4,  9): 12.8, datetime.date(2025,  4, 10): 13.6,
    datetime.date(2025,  4, 11): 19.1, datetime.date(2025,  4, 12): 20.2,
    datetime.date(2025,  4, 13): 14.3, datetime.date(2025,  4, 14): 13.0,
    datetime.date(2025,  4, 15): 14.9, datetime.date(2025,  4, 16): 13.4,
    datetime.date(2025,  4, 17): 15.2, datetime.date(2025,  4, 18): 14.6,
    datetime.date(2025,  4, 19): 15.6, datetime.date(2025,  4, 20): 15.8,
    datetime.date(2025,  4, 21): 14.1, datetime.date(2025,  4, 22): 15.4,
    datetime.date(2025,  4, 23): 13.6, datetime.date(2025,  4, 24): 13.8,
    datetime.date(2025,  4, 25): 14.4, datetime.date(2025,  4, 26): 15.9,
    datetime.date(2025,  4, 27): 19.1, datetime.date(2025,  4, 28): 20.5,
    datetime.date(2025,  4, 29): 22.8, datetime.date(2025,  4, 30): 24.1,
    datetime.date(2025,  5,  1): 26.4, datetime.date(2025,  5,  2): 22.6,
    datetime.date(2025,  5,  3): 17.5, datetime.date(2025,  5,  4): 13.1,
    datetime.date(2025,  5,  5): 13.0, datetime.date(2025,  5,  6): 13.4,
    datetime.date(2025,  5,  7): 16.8, datetime.date(2025,  5,  8): 13.8,
    datetime.date(2025,  5,  9): 18.2, datetime.date(2025,  5, 10): 21.8,
    datetime.date(2025,  5, 11): 23.1, datetime.date(2025,  5, 12): 20.9,
    datetime.date(2025,  5, 13): 22.8, datetime.date(2025,  5, 14): 20.6,
    datetime.date(2025,  5, 15): 15.8, datetime.date(2025,  5, 16): 19.3,
    datetime.date(2025,  5, 17): 16.6, datetime.date(2025,  5, 18): 16.9,
    datetime.date(2025,  5, 19): 17.9, datetime.date(2025,  5, 20): 20.7,
    datetime.date(2025,  5, 21): 18.6, datetime.date(2025,  5, 22): 15.1,
    datetime.date(2025,  5, 23): 19.0, datetime.date(2025,  5, 24): 19.7,
    datetime.date(2025,  5, 25): 18.3, datetime.date(2025,  5, 26): 17.2,
    datetime.date(2025,  5, 27): 16.3, datetime.date(2025,  5, 28): 19.5,
    datetime.date(2025,  5, 29): 22.4, datetime.date(2025,  5, 30): 22.7,
    datetime.date(2025,  5, 31): 24.1,
    datetime.date(2025,  6,  1): 18.7, datetime.date(2025,  6,  2): 20.0,
    datetime.date(2025,  6,  3): 18.1, datetime.date(2025,  6,  4): 17.5,
    datetime.date(2025,  6,  5): 16.6, datetime.date(2025,  6,  6): 17.1,
    datetime.date(2025,  6,  7): 16.0, datetime.date(2025,  6,  8): 16.2,
    datetime.date(2025,  6,  9): 18.3, datetime.date(2025,  6, 10): 21.0,
    datetime.date(2025,  6, 11): 22.6, datetime.date(2025,  6, 12): 23.0,
    datetime.date(2025,  6, 13): 25.9, datetime.date(2025,  6, 14): 21.2,
    datetime.date(2025,  6, 15): 21.8, datetime.date(2025,  6, 16): 23.5,
    datetime.date(2025,  6, 17): 24.0, datetime.date(2025,  6, 18): 27.0,
    datetime.date(2025,  6, 19): 29.4, datetime.date(2025,  6, 20): 28.5,
    datetime.date(2025,  6, 21): 29.5, datetime.date(2025,  6, 22): 23.5,
    datetime.date(2025,  6, 23): 21.2, datetime.date(2025,  6, 24): 23.7,
    datetime.date(2025,  6, 25): 25.7, datetime.date(2025,  6, 26): 22.0,
    datetime.date(2025,  6, 27): 27.4, datetime.date(2025,  6, 28): 27.0,
    datetime.date(2025,  6, 29): 27.1, datetime.date(2025,  6, 30): 30.4,
    datetime.date(2025,  7,  1): 29.5, datetime.date(2025,  7,  2): 23.3,
    datetime.date(2025,  7,  3): 21.6, datetime.date(2025,  7,  4): 24.6,
    datetime.date(2025,  7,  5): 22.0, datetime.date(2025,  7,  6): 21.5,
    datetime.date(2025,  7,  7): 19.8, datetime.date(2025,  7,  8): 22.4,
    datetime.date(2025,  7,  9): 25.8, datetime.date(2025,  7, 10): 29.0,
    datetime.date(2025,  7, 11): 31.2, datetime.date(2025,  7, 12): 29.1,
    datetime.date(2025,  7, 13): 26.4, datetime.date(2025,  7, 14): 23.7,
    datetime.date(2025,  7, 15): 20.8, datetime.date(2025,  7, 16): 24.2,
    datetime.date(2025,  7, 17): 25.3, datetime.date(2025,  7, 18): 25.9,
    datetime.date(2025,  7, 19): 21.9, datetime.date(2025,  7, 20): 20.9,
    datetime.date(2025,  7, 21): 22.5, datetime.date(2025,  7, 22): 21.1,
    datetime.date(2025,  7, 23): 20.4, datetime.date(2025,  7, 24): 22.0,
    datetime.date(2025,  7, 25): 25.2, datetime.date(2025,  7, 26): 23.1,
    datetime.date(2025,  7, 27): 20.4, datetime.date(2025,  7, 28): 22.3,
    datetime.date(2025,  7, 29): 20.8, datetime.date(2025,  7, 30): 23.5,
    datetime.date(2025,  7, 31): 21.8,
    datetime.date(2025,  8,  1): 20.0, datetime.date(2025,  8,  2): 21.5,
    datetime.date(2025,  8,  3): 23.8, datetime.date(2025,  8,  4): 21.7,
    datetime.date(2025,  8,  5): 19.9, datetime.date(2025,  8,  6): 22.2,
    datetime.date(2025,  8,  7): 23.6, datetime.date(2025,  8,  8): 23.2,
    datetime.date(2025,  8,  9): 24.6, datetime.date(2025,  8, 10): 25.5,
    datetime.date(2025,  8, 11): 28.3, datetime.date(2025,  8, 12): 31.9,
    datetime.date(2025,  8, 13): 27.0, datetime.date(2025,  8, 14): 25.1,
    datetime.date(2025,  8, 15): 28.3, datetime.date(2025,  8, 16): 24.2,
    datetime.date(2025,  8, 17): 25.2, datetime.date(2025,  8, 18): 22.6,
    datetime.date(2025,  8, 19): 23.1, datetime.date(2025,  8, 20): 19.2,
    datetime.date(2025,  8, 21): 19.7, datetime.date(2025,  8, 22): 23.5,
    datetime.date(2025,  8, 23): 21.8, datetime.date(2025,  8, 24): 24.1,
    datetime.date(2025,  8, 25): 27.1, datetime.date(2025,  8, 26): 24.0,
    datetime.date(2025,  8, 27): 20.7, datetime.date(2025,  8, 28): 20.9,
    datetime.date(2025,  8, 29): 19.6, datetime.date(2025,  8, 30): 19.8,
    datetime.date(2025,  8, 31): 19.8,
    datetime.date(2025,  9,  1): 20.0, datetime.date(2025,  9,  2): 18.8,
    datetime.date(2025,  9,  3): 18.8, datetime.date(2025,  9,  4): 18.4,
    datetime.date(2025,  9,  5): 20.3, datetime.date(2025,  9,  6): 21.9,
    datetime.date(2025,  9,  7): 22.8, datetime.date(2025,  9,  8): 19.2,
    datetime.date(2025,  9,  9): 19.7, datetime.date(2025,  9, 10): 18.1,
    datetime.date(2025,  9, 11): 16.4, datetime.date(2025,  9, 12): 16.6,
    datetime.date(2025,  9, 13): 16.8, datetime.date(2025,  9, 14): 17.0,
    datetime.date(2025,  9, 15): 16.8, datetime.date(2025,  9, 16): 17.8,
    datetime.date(2025,  9, 17): 18.8, datetime.date(2025,  9, 18): 21.8,
    datetime.date(2025,  9, 19): 24.2, datetime.date(2025,  9, 20): 20.5,
    datetime.date(2025,  9, 21): 14.6, datetime.date(2025,  9, 22): 14.9,
    datetime.date(2025,  9, 23): 16.2, datetime.date(2025,  9, 24): 16.8,
    datetime.date(2025,  9, 25): 18.0, datetime.date(2025,  9, 26): 15.8,
    datetime.date(2025,  9, 27): 16.2, datetime.date(2025,  9, 28): 17.3,
    datetime.date(2025,  9, 29): 17.4, datetime.date(2025,  9, 30): 18.2,
    datetime.date(2025, 10,  1): 19.1, datetime.date(2025, 10,  2): 17.8,
    datetime.date(2025, 10,  3): 17.3, datetime.date(2025, 10,  4): 13.8,
    datetime.date(2025, 10,  5): 15.4, datetime.date(2025, 10,  6): 18.6,
    datetime.date(2025, 10,  7): 16.9, datetime.date(2025, 10,  8): 16.9,
    datetime.date(2025, 10,  9): 15.0, datetime.date(2025, 10, 10): 15.9,
    datetime.date(2025, 10, 11): 14.2, datetime.date(2025, 10, 12): 13.1,
    datetime.date(2025, 10, 13): 14.9, datetime.date(2025, 10, 14): 13.8,
    datetime.date(2025, 10, 15): 11.9, datetime.date(2025, 10, 16): 14.2,
    datetime.date(2025, 10, 17): 14.4, datetime.date(2025, 10, 18): 14.4,
    datetime.date(2025, 10, 19): 14.9, datetime.date(2025, 10, 20): 15.2,
    datetime.date(2025, 10, 21): 14.4, datetime.date(2025, 10, 22): 14.1,
    datetime.date(2025, 10, 23): 12.1, datetime.date(2025, 10, 24): 11.5,
    datetime.date(2025, 10, 25): 10.6, datetime.date(2025, 10, 26): 11.5,
    datetime.date(2025, 10, 27): 13.0, datetime.date(2025, 10, 28): 13.8,
    datetime.date(2025, 10, 29): 11.6, datetime.date(2025, 10, 30): 13.2,
    datetime.date(2025, 10, 31): 14.8,
    datetime.date(2025, 11,  1): 13.9, datetime.date(2025, 11,  2): 11.9,
    datetime.date(2025, 11,  3): 15.8, datetime.date(2025, 11,  4): 15.9,
    datetime.date(2025, 11,  5): 16.7, datetime.date(2025, 11,  6): 16.7,
    datetime.date(2025, 11,  7): 15.2, datetime.date(2025, 11,  8): 14.1,
    datetime.date(2025, 11,  9): 13.4, datetime.date(2025, 11, 10): 12.7,
    datetime.date(2025, 11, 11): 14.5, datetime.date(2025, 11, 12): 15.2,
    datetime.date(2025, 11, 13): 15.8, datetime.date(2025, 11, 14): 12.1,
    datetime.date(2025, 11, 15): 12.8, datetime.date(2025, 11, 16):  9.8,
    datetime.date(2025, 11, 17):  6.9, datetime.date(2025, 11, 18):  7.2,
    datetime.date(2025, 11, 19):  4.7, datetime.date(2025, 11, 20):  3.8,
    datetime.date(2025, 11, 21):  5.1, datetime.date(2025, 11, 22):  8.1,
    datetime.date(2025, 11, 23):  9.4, datetime.date(2025, 11, 24):  7.1,
    datetime.date(2025, 11, 25):  6.4, datetime.date(2025, 11, 26):  9.1,
    datetime.date(2025, 11, 27): 13.4, datetime.date(2025, 11, 28): 11.9,
    datetime.date(2025, 11, 29): 10.1, datetime.date(2025, 11, 30):  7.4,
    datetime.date(2025, 12,  1): 12.1, datetime.date(2025, 12,  2): 10.1,
    datetime.date(2025, 12,  3):  8.6, datetime.date(2025, 12,  4):  7.3,
    datetime.date(2025, 12,  5):  9.4, datetime.date(2025, 12,  6): 11.4,
    datetime.date(2025, 12,  7): 13.9, datetime.date(2025, 12,  8): 13.1,
    datetime.date(2025, 12,  9): 14.8, datetime.date(2025, 12, 10): 11.8,
    datetime.date(2025, 12, 11): 11.4, datetime.date(2025, 12, 12): 11.2,
    datetime.date(2025, 12, 13):  8.8, datetime.date(2025, 12, 14): 10.9,
    datetime.date(2025, 12, 15): 11.7, datetime.date(2025, 12, 16):  9.7,
    datetime.date(2025, 12, 17): 10.6, datetime.date(2025, 12, 18): 11.5,
    datetime.date(2025, 12, 19):  9.7, datetime.date(2025, 12, 20):  8.4,
    datetime.date(2025, 12, 21): 10.1, datetime.date(2025, 12, 22): 10.9,
    datetime.date(2025, 12, 23):  7.6, datetime.date(2025, 12, 24):  5.4,
    datetime.date(2025, 12, 25):  4.6, datetime.date(2025, 12, 26):  5.2,
    datetime.date(2025, 12, 27):  6.6, datetime.date(2025, 12, 28):  6.6,
    datetime.date(2025, 12, 29):  5.7, datetime.date(2025, 12, 30):  5.3,
    datetime.date(2025, 12, 31):  2.8,
}

TMIN = 0    # °C, colormap lower bound
TMAX = 32   # °C, colormap upper bound

WARM_DAY = "▟█"
COOL_DAY = "▟█"
NOT_A_DAY = "  "


def main(save: str | None = None):
    """Calendar heatmap of daily max temperatures, Oxford 2025."""
    month_plots = []
    for month in range(1, 13):
        year = 2025
        title = mp.text(f"{calendar.month_name[month]:<9s} {year:4d}")
        daynames = mp.text("M T W t F S s ")
        week_plots = []
        for week in calendar.monthcalendar(year, month):
            day_plots = []
            for day in week:
                if day == 0:
                    day_plots.append(mp.text(NOT_A_DAY))
                    continue
                date = datetime.date(year, month, day)
                t = DATA[date]
                frac = max(0.0, min(1.0, (t - TMIN) / (TMAX - TMIN)))
                day_plots.append(mp.text(
                    WARM_DAY,
                    fgcolor=mp.inferno(frac),
                    bgcolor=(0, 0, 0),
                ))
            week_plots.append(mp.hstack(*day_plots))
        month_plots.append(
            mp.vstack(title, daynames, *week_plots)
            + mp.blank(2, 2),
        )

    plot = mp.wrap(*month_plots, cols=4)

    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    tyro.cli(main)
