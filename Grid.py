
class Grid:

    def __init__(self):
        self.rows = set()
        self.cols = set()
        self.data = {}

    def set(self, row_key, col_key, value):
        self.rows.add(row_key)
        self.cols.add(col_key)
        if row_key not in self.data:
            self.data[row_key] = {}
        if col_key not in self.data[row_key]:
            self.data[row_key][col_key] = {}
        self.data[row_key][col_key] = value

    def get(self, row_key, col_key):
        return None

    def load(self, file_name):
        pass

    def save(self, file_name):
        cols = sorted(self.cols)
        rows = sorted(self.rows)
        with open(file_name, "w") as outfile:
            outfile.write("\t{}\n".format('\t'.join(cols)))
            for row_key in rows:
                outfile.write(row_key)
                for col_key in cols:
                    if row_key in self.data and col_key in self.data[row_key]:
                        outfile.write("\t{}".format(self.data[row_key][col_key]))
                    else:
                        outfile.write("\t")
                outfile.write("\n")
