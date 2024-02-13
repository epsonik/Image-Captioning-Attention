def parse_file(filename):
    lines = []
    lines_dict = []
    index = 0
    with open(filename) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith(' * LOSS - '):
                a = line.split(',')
                loss = a[0].split(" - ")
                accuracy = a[1].split(" - ")
                bleu_4 = a[2].split(" - ")
                cider = a[3].split(" - ")
                lines_dict.append(
                    {"index": index, "loss": loss[1], "accuracy": accuracy[1], "bleu_4": bleu_4[1], "cider": cider[1],
                     "line": line})
                lines.append(line)
                index += 1

    return lines_dict


# * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}, CIDEr - {cider}

if __name__ == '__main__':
    lines = parse_file("test_parse.txt")
    for line in lines:
        print(line)
