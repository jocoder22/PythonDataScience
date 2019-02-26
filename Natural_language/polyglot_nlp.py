import polyglot
from polyglot.text import Text, Word


parts_by_year = sets[['year', 'num_parts']].groupby(
    'year', as_index=False).mean()
# parts_by_year
# Plot trends in average number of parts by year
parts_by_year.plot(x='year', y='num_parts')


colors_summary = colors.groupby('is_trans', as_index=False).count()
