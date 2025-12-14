from SoccerNet.Downloader import SoccerNetDownloader

# For labels only (no password)
downloader = SoccerNetDownloader(LocalDirectory='./SoccerNet-Ball2024')
downloader.downloadDataTask(task='spotting-ball-2024', split=['valid'])

# For videos (requires password)
password = input('Enter SoccerNet password: ')
downloader.password = password
downloader.downloadDataTask(task='spotting-ball-2024', split=['valid'], password=password)

print('✅ Download complete!')
