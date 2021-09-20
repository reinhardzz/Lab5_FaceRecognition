#Cosine distance matrix
def cosine_distance(A, B):
  A_B = np.sum(A * B)
  abs_A = math.sqrt(np.sum(A ** 2))
  abs_B = math.sqrt(np.sum(B ** 2))
  cosine_similarity = A_B / (abs_A * abs_B)
  return 1 - cosine_similarity

distanceList =[]
# Cosine distance
SampleID = 0
for i in closedEnrFaceFocusFaceEncode :
  distance = []
  SampleID = SampleID +1
  for j in closedEnrRegister:
    distance.append(cosine_distance(i,j[1]))
  distanceList.append([SampleID,closedEnrFaceName[SampleID-1]] + distance)
df = pd.DataFrame(np.array(distanceList),columns=["SampleID","ActualID"] + closedEnrDsName)
df