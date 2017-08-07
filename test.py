def main():
	log('It worked here.')

def log(msg):
    
    msg = str(msg)
    path = '/cluster/home/paulusm/data/beer_reviews/test_file.txt'

    with open(path, 'a') as log:
        log.write(msg + '\n')

if __name__ == '__main__':
    main()