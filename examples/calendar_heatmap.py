"""
GitHub-inspired calendar visualisation.
"""

import matthewplotlib as mp
import calendar
import datetime


DATA = {
    datetime.date(2024,  5,  1): 1,
    datetime.date(2025,  4,  1): 7,
    datetime.date(2025,  4,  2): 9,
    datetime.date(2025,  4,  3): 1,
    datetime.date(2025,  4,  4): 6,
    datetime.date(2025,  4,  5): 2,
    datetime.date(2025,  4,  6): 8,
    datetime.date(2025,  4,  7): 5,
    datetime.date(2025,  4,  8): 7,
    datetime.date(2025,  4,  9): 0,
    datetime.date(2025,  4, 10): 7,
    datetime.date(2025,  4, 11): 4,
    datetime.date(2025,  4, 12): 8,
    datetime.date(2025,  4, 13): 1,
    datetime.date(2025,  4, 14): 8,
    datetime.date(2025,  4, 15): 1,
    datetime.date(2025,  4, 16): 9,
    datetime.date(2025,  4, 17): 1,
    datetime.date(2025,  4, 18): 9,
    datetime.date(2025,  4, 19): 1,
    datetime.date(2025,  4, 20): 1,
    datetime.date(2025,  4, 21): 8,
    datetime.date(2025,  4, 22): 7,
    datetime.date(2025,  4, 23): 7,
    datetime.date(2025,  4, 24): 1,
    datetime.date(2025,  4, 25): 6,
    datetime.date(2025,  4, 26): 4,
    datetime.date(2025,  4, 27): 5,
    datetime.date(2025,  4, 28): 0,
    datetime.date(2025,  4, 29): 3,
    datetime.date(2025,  4, 30): 0,
    datetime.date(2025,  5,  1): 5,
    datetime.date(2025,  5,  2): 10,
    datetime.date(2025,  5,  3): 0,
    datetime.date(2025,  5,  4): 5,
    datetime.date(2025,  5,  5): 10,
    datetime.date(2025,  5,  6): 7,
    datetime.date(2025,  5,  7): 4,
    datetime.date(2025,  5,  8): 8,
    datetime.date(2025,  5,  9): 2,
    datetime.date(2025,  5, 10): 4,
    datetime.date(2025,  5, 11): 6,
    datetime.date(2025,  5, 12): 3,
    datetime.date(2025,  5, 13): 3,
    datetime.date(2025,  5, 14): 2,
    datetime.date(2025,  5, 15): 8,
}


COUNT_DAY = "▟█"
EMPTY_DAY = "▘ "
NOT_A_DAY = "  "



def main():
    # normalise counts
    max_count = max(DATA.values())
    norm_data = {date: count/max_count for date, count in DATA.items()}

    start_date = min(DATA.keys())
    end_date = max(DATA.keys())
    year = start_date.year
    month = start_date.month
    month_plots = []
    while datetime.date(year, month, 1) <= end_date:
        # collect month
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
                if date not in DATA:
                    day_plots.append(mp.text(EMPTY_DAY, color=(0,0,0)))
                    continue
                day_plots.append(mp.text(
                    COUNT_DAY,
                    color=mp.cool(1-norm_data[date]),
                    bgcolor=(0,0,0),
                ))
            week_plots.append(mp.hstack(*day_plots))
        month_plots.append(
            mp.vstack(title, daynames, *week_plots)
            | mp.blank(2,2),
        )
        
        # increment month
        month += 1
        if month == 13:
            year += 1
            month = 1

    plot = mp.wrap(*month_plots)
    
    print("printing plot...")
    print(plot)
    print("saving to 'out.png'...")
    plot.saveimg('out.png')


if __name__ == "__main__":
    main()

