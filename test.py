def main():
    print("Executing.")
    log('It worked here.')
    print("Executed.")

def log(msg):
    
    msg = str(msg)
    path = '/cluster/home/paulusm/data/beer_reviews/test_file.txt'

    with open(path, 'a') as log:
        log.write(msg)

if __name__ == '__main__':
    main()