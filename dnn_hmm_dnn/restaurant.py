EPS = 1e-100
class Restaurant:
  # Attributes:
  # ----------
  #   tables: a list [count_1, ..., count_T], 
  #           where count_t is the number of customers with at table t;
  #   name2table: a dictionary {k:t}, mapping name k to table t
  #   ncustomers: sum(tables),
  #               storing the total number of customers with each dish; 
  #   ntables: len(tables),
  #            total number of tables;
  #   p_init: a dictionary {k: p_0(k)},
  #         where p_0(k) is the initial probability for table with name k
  #   alpha0: concentration, Dirichlet process parameter
  def __init__(self, alpha0):
    self.tables = []
    self.ntables = 0
    self.ncustomers = 0
    self.name2table = {}
    self.table_names = []
    self.p_init = {}
    self.alpha0 = alpha0

  def seat_to(self, k, w=1):
    self.ncustomers += 1 
    tables = self.tables # shallow copy the tables to a local variable
    if not k in self.name2table: # add a new table
      tables.append(w)
      self.name2table[k] = self.ntables
      self.table_names.append(k)
      self.ntables += 1
    else:
      i = self.name2table[k]
      tables[i] += w

  def unseat_from(self, k, w=1):
    self.ncustomers -= 1
    i = self.name2table[k]
    tables = self.tables
    tables[i] -= w
    if tables[i] <= EPS: # cleanup empty table
      k_new = self.table_names[-1] 
      self.table_names[i] = k_new # replace the empty table with the last table
      self.name2table[k_new] = i
      self.tables[i] = self.tables[-1]
      del self.name2table[k] 
      del self.table_names[-1]
      del self.tables[-1]
      self.ntables -= 1 

  def prob(self, k, p_init=None):
    if not p_init:
      p_init = self.p_init[k]
    else:
      self.p_init[k] = p_init

    w = self.alpha0 * p_init 
    if k in self.name2table:
      i = self.name2table[k]
      w += self.tables[i]
    
    return w / (self.alpha0 + self.ncustomers) 

  def log_likelihood(self):
    ll = math.lgamma(self.alpha0) - math.lgamma(self.alpha0 + self.ncustomers)
    ll += sum(math.lgamma(self.tables[i] + self.alpha0 * self.p_init[k]) for i, k in enumerate(self.table_names))
    ll += sum(self.p_init[k] - math.lgamma(self.alpha0 * self.p_init[k]) for k in self.table_names)
    return ll

  def save(self, outputDir='./'):
    with open(outputDir + 'tables.txt', 'w') as f:
      sorted_indices = sorted(list(range(self.ntables)), key=lambda x:self.tables[x], reverse=True)
      for i in sorted_indices:
        f.write('%s %.5f\n' % (self.table_names[i], self.tables[i]))
