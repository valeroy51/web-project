from datetime import datetime
from babel.dates import format_date

tanggal = datetime.now()

hasil = format_date(
    tanggal,
    format="EEEE, d MMMM y",
    locale="id"
)

print(hasil)
# Senin, 6 Januari 2026
