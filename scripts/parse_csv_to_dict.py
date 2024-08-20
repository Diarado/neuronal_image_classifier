import csv

def parse_csv_to_dict(csv_path):
    data_dict = {}

    with open(csv_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            #print(row)
            # Skip empty rows
            if not any(row):
                continue

            plate_info = row[0].strip()
            #print(plate_info)
            scores = row[1:]  # All the scores after plate_info
            #print(scores)
            fields = []

            # Iterate over scores 4 at a time for each field
            for i in range(0, len(scores), 4):
                field_scores = scores[i:i+4]
                #print(field_scores)

                peeling, contaminants, cell_density = field_scores[:3]
                if field_scores[3]: # if there is some string in empty/dead, we make dead True
                    empty_dead = True

                else:
                    empty_dead = False
                    
                peel_score = int(round(float(peeling))) if peeling else 1
                cont_score = int(round(float(contaminants))) if contaminants else 1
                den_score = int(round(float(cell_density))) if cell_density else 1
                fields.append([peel_score, cont_score, den_score, empty_dead])
                
            # Store the list of fields for the plate_info key
            data_dict[plate_info] = fields
    # print(data_dict)
    return data_dict

# parse_csv_to_dict('train/scoring_round06.csv')