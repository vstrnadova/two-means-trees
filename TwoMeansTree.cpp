#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#include <queue>
#include <cstdlib>
#include <assert.h>
#include <algorithm> //sort
#include <limits>
#include <numeric> //accumulate
#include <string.h> //strcmp
#include <memory> //unique_ptr
#include <unordered_map>
#include <utility> //pair
#include <sys/time.h>
#include <boost/algorithm/string.hpp>

using namespace std;

/* store pairs of indices (i,j) for each pair 
of vertices that occur in a leaf node, and the number
of times they co-occur */ 

map< pair<int, int>, double > coOccurMap;

bool greaterSecondInPair( pair<int, double> a, pair<int, double> b){
    return a.second > b.second;
}

bool secondInPair( pair<int, double> a, pair<int, double> b){
    return a.second < b.second;
}

class TwoMeansTreeNode {
	private:
		TwoMeansTreeNode *left;
		TwoMeansTreeNode *right;
		unsigned int depth;
		double midpoint;
		int splitDim;
		//vector< double > means;
		//double means[2];
		vector< vector<double> > points;
        vector<int> pointIndices;
		int ndimensions;
		bool leafNode;
		int ID;
	public: 
		TwoMeansTreeNode(vector< vector<double> > pts, vector<int> ptIdxs, unsigned int d, bool isLeafNode, int id){
			if(isLeafNode){
			    //set points at this node
			    for(int i=0; i<pts.size();i++){
			    	points.push_back(pts[i]);
			    }
                for(int i=0; i<pts.size(); i++){
                    pointIndices.push_back(ptIdxs[i]);
                }
			    leafNode = true;
			    ID = id;
			    //cout << "added points to leaf node"<<endl;
			} else {
				leafNode = false;
				assert(pts.size() == 1);
				if((pts[0]).size() != 1){
					//cout << "(pts[0]).size() = "<<(pts[0]).size()<<endl;
					//cout << "Internal node only stores one point, the midpoint. "<<endl;
				}
				midpoint = pts[0][0];
				splitDim = -1;
				ID = id;
			}
			left = NULL;
			right = NULL;
			depth = d;
		} 
		
		~TwoMeansTreeNode(){
			delete this->right;
			delete this->left;
		}
		
		void setRightChild(TwoMeansTreeNode* t1){
			this->right = t1;
		}
	
		void setLeftChild(TwoMeansTreeNode* t2){
			this->left = t2;
		}
	
		void setSplitDim(int splitdim){
			this->splitDim = splitdim;
		}
		
		int getSplitDim(){
			return this->splitDim;
		}
		
		unsigned int getDepth(){
			return this->depth;
		}
		
		TwoMeansTreeNode* getLeftChild(){
			return this->left;
		}
		
		TwoMeansTreeNode* getRightChild(){
			return this->right;
		}
	
		vector< vector<double> > getPoints(){
			return this->points;
		}
		
		double getMidpoint(){
			return this->midpoint;
		}
		
		bool isLeafNode(){
			return this->leafNode;
		}
		
		int getID(){
			return this->ID;
		}
    
        vector<int> getPointIndices(){
            return this->pointIndices;
        }
};

bool occurCountGreaterThan(pair< vector<double>, int > a, pair< vector<double>, int > b){
	return a.second > b.second;
}

void printCoOccurMap(int datasetsize){
	stringstream ofss;
	ofss<<"coOccurMap_"<<datasetsize<<"_pts.txt";
	cout << "printing leaf coOccurMap to file: "<<ofss.str()<<endl;
	ofstream coOccur_file;
	coOccur_file.open(ofss.str().c_str());
    for(map<pair<int, int>, double>::iterator iter = coOccurMap.begin(); iter != coOccurMap.end(); ++iter)
    {
        pair<int,int> pairkey =  iter->first;
        coOccur_file << "("<< pairkey.first <<" , "<<pairkey.second<<") : " << iter->second << endl;
    }

    /*for(auto i = 0; i<datasetsize; i++)
	{	
		auto found = coOccurMap.find (i);
		if(found != coOccurMap.end()){
			auto pairvec = coOccurMap.at(i);
			coOccur_file << i <<endl;
			for(int j=0; j<pairvec.size(); j++){
				pair< int, int > countpair = pairvec[j];
				coOccur_file << " ("<<countpair.first<<", "<<countpair.second<<"), ";
			}
			coOccur_file <<endl;
		}
	}*/
}

void printLevelOrder(TwoMeansTreeNode *tnode)
{
    // Base Case
    if (tnode == NULL)  return;
 
    // Create an empty queue for level order tarversal
    queue<TwoMeansTreeNode *> q;
 
    // Enqueue Root and initialize height
    q.push(tnode);
 
    while (q.empty() == false)
    {
        TwoMeansTreeNode *node = q.front();
        if(node->isLeafNode()){
		cout << " [num. pts ="<< (node->getPoints()).size() <<" ] ";
        } else {
		cout << " [midpoint at depth "<<node->getDepth()<<":  "<<node->getMidpoint()<<"] ";
	}
	q.pop();
 
        /* Enqueue left child */
        if (node->getLeftChild() != NULL)
            q.push(node->getLeftChild());
 
        /*Enqueue right child */
        if (node->getRightChild() != NULL)
            q.push(node->getRightChild());
    }
}
 

void printTree(TwoMeansTreeNode *tree){
	assert( (tree->getLeftChild()== NULL && tree->getRightChild() == NULL ) || (tree->getLeftChild()!= NULL && tree->getRightChild() != NULL));
	if( tree->getLeftChild() == NULL && tree->getRightChild()==NULL ){
		vector< vector<double> > pts = tree->getPoints();
		if(pts.size()==0) cout << "Warning: Leaf node with no points."<<endl;
		cout << "Number of leaf node "//<< tree->getID() 
			<<"points at depth "<<tree->getDepth() <<": "<<pts.size()<<endl;
	} else {
		cout <<"Internal node "//<< tree->getID() 
			<<" at depth " << tree->getDepth(); 
			//<<" midpoint: "<<tree->getMidpoint()<<endl;
			printf("midpoint: %f10 \nsplitting dimension: %d",  tree->getMidpoint(), tree->getSplitDim() );
		cout<<endl;
		unsigned int depthcurrent = tree->getDepth();
		if(depthcurrent==0){
			cout << depthcurrent<<"--";
		}
		printTree(tree->getLeftChild());
		printTree(tree->getRightChild());		
	}
	return;
}

double euclideanDistance(vector< double > a, vector< double > b){
	assert(a.size() == b.size());
	vector<double> sqdists(a.size(), 0.0);
	for(int i=0; i<a.size(); i++){
		sqdists[i] = (a[i]-b[i])*(a[i]-b[i]);
	}
	double dist = sqrt(accumulate(sqdists.begin(), sqdists.end(), 0.0));
	return dist;
}

void printLeafNodes(TwoMeansTreeNode *tree){	
	if( tree->getLeftChild() == NULL && tree->getRightChild()==NULL ){
		vector< vector<double> > pts = tree->getPoints();
		cout << "Leaf node "//<< tree->getID() 
			<<"points:\n "<<endl;
		for(int i=0; i<pts.size(); i++){
			cout << "(";
			for(int j=0; j<(pts[i]).size(); j++){
				if(j<(pts[i]).size()-1){
					cout << pts[i][j] <<",";
				} else {
					cout << pts[i][j] <<")";
				}
			}
			if(i<pts.size()-1){
				cout << ", ";		
			}
		}
		cout <<endl;
	} else {
		printLeafNodes(tree->getLeftChild());
		printLeafNodes(tree->getRightChild());		
	}
	return;
}

/* twoMeansOneD: 
 *	finds the optimal 2-means clustering 
 * 	of data in one dimension
 *	returns a pair that holds the
 * 	sum of squared distances from points
 *	in each cluster to their
 *	means, and the midpoint
 *	between the boundaries of the two
 *	clusters  
 */
pair< double, double > twoMeansOneD(vector<double> &X){
	int npts = X.size();
	//cout << "twoMeansOneD: number of input points = "<<npts<<endl;
 	
	/* process points in sorted order */	
	sort(X.begin(), X.end());
	double midpoint;
	vector<double> means;
	vector<double>::iterator it=X.begin();
	//initialize the two means
	double mean1=0.0, mean2=0.0, sum1=0.0, sum2=0.0;
	vector<double>::iterator it2 = X.begin();
	while(it2<X.end()){
		sum2 += *it2;
		it2++;
	}
    
	mean2 = sum2/(double) npts;
    
	/* initially, set2 contains all points
	 *	and set1 contains no points 	*/
	int seenpts=0;
	double sumsqdists=0.0, minsumsqdists=numeric_limits<double>::max();
	while(seenpts<(npts-1)){
		//if(npts<4) cout << "seen pts = "<<seenpts<<endl;
		seenpts++;
		/* adjust sums */
		sum1+=*it;
		sum2-=*it;
		/* adjust means */
		mean1 = sum1/(double) seenpts;
		//if(npts<4) cout << "mean1 = "<<mean1<<endl;
		mean2 = sum2/(double) (npts-seenpts);
		//if(npts<4) cout << "mean2 = "<<mean2<<endl;
		//cout << "iteration "<<seenpts<<": means found: mean1="<<mean1<<", mean2="<<mean2<<endl;
		//cout << " sum1="<<sum1<<", sum2="<<sum2<<endl;	
		
	
		/* assign points to closest mean and update sumsqdists */
		sumsqdists=0.0;
		for(int i=0; i<npts; i++){
			double dist1 = fabs(X[i] - mean1);
			double dist2 = fabs(X[i] - mean2);
			if(dist1<dist2){
				sumsqdists += dist1*dist1;
			} else {
			   	sumsqdists += dist2*dist2;
			}
		}
	
		/* store means if sum of squared distances
		* to means yields best split
		*/
		if(sumsqdists<minsumsqdists){
			minsumsqdists = sumsqdists;
			double meanofmeans = (mean1+mean2)/2;
			midpoint = (*it+*(it+1))/2.0; //midpoint is midpoint between the borders of the clusters represented by the two means
			/*if(npts<4) cout << "midpoint = "<<midpoint<<endl;
			cout << "min sum sq. dists = "<<minsumsqdists<<endl;
			cout << "mean1 = "<<mean1;
			cout << ", mean2 = "<<mean2<<endl;
			cout << "midpoint = "<<midpoint<<endl;
			cout << "mean of means = "<<meanofmeans<<endl;
			*/
		}
		it++;
	}	
	//cout << "twoMeansOneD: returning midpont " <<midpoint<<endl;
	return make_pair(minsumsqdists, midpoint);
}

/* perform 2-means clustering on a set of 
	input data X
	return the two means 
*/
vector< vector<double> > twomeans(vector< vector<double> > &X){
	int npts = X.size();
	int ndims=0;
	if(!X.empty()){
		ndims = (X[0]).size();
	} 
	bool isCloserToMean1[npts];	
	//choose starting means at random from the points in the data set
	int idx1 = rand() % npts;
	int idx2 = rand() % npts;	
	vector< double > mean1 = X[idx1];
	vector< double > mean2 = X[idx2];
	//keep track of how many swaps happened 
	// between this iteration and the last
	int nSwaps = npts;
	int iters=0, maxiterations=1000;
	while (nSwaps>0 && iters<maxiterations){ //iterate until no points are re-assigned or maximum number of iterations is reached
		//iteratively update means and re-assign points
		nSwaps = 0;
		iters++;
		// assign points to closest mean
		for(int i=0; i<npts; i++){
			double dist1 = euclideanDistance(X[i], mean1);
			double dist2 = euclideanDistance(X[i], mean2);
			if(dist1<dist2){
				if(isCloserToMean1[i] == false) nSwaps++;
				isCloserToMean1[i] = true;
			} else {
				if(isCloserToMean1[i] == true) nSwaps++;
				isCloserToMean1[i] = false;	
			}
		}
		// re-compute means
		mean1.clear();
		mean1.resize(ndims, 0.0);
		mean2.clear();
		mean2.resize(ndims, 0.0);
		double npts1 = 0.0, npts2 = 0.0;
		for(int i=0; i<npts; i++){
			if(isCloserToMean1[i]){ 
				for(int dim=0; dim<ndims; dim++){
					mean1[dim] += (X[i])[dim];
				}
				npts1++;
			} else { 
				for(int dim=0; dim<ndims; dim++){
					mean2[dim] += (X[i])[dim];
				}
				npts2++;
			}
		}
		for(int dim=0; dim<ndims; dim++){
			mean1[dim] = mean1[dim]/npts1;
		}
		for(int dim=0; dim<ndims; dim++){
			mean2[dim] = mean2[dim]/npts2;
		}
	}
	vector< vector<double> > means;
	means.push_back(mean1);
	means.push_back(mean2);
	return means;
}
	
void split_1D_DataBy2Means(vector<vector<double> > X, double midpt, bool *closerToMu1, int splitting_dimension){
	int n = X.size();
	int ndims=0;
	if(!X.empty()){
		ndims = X[0].size();
	}
	for(int i=0; i<n; i++){
		if(X[i][splitting_dimension] < midpt){
			closerToMu1[i] = true;	
		} else { // in this case d_mu1 >= d_mu2
			closerToMu1[i] = false;
		} 
	}
}

void splitDataBy2Means(vector<vector<double> > X, vector< vector<double> > mus, bool *closerToMu1){
	int n = X.size();
	int ndims=0;
	if(!X.empty()){
		ndims = X[0].size();
	}
	double d_mu1, d_mu2;
	for(int i=0; i<n; i++){
		d_mu1 = euclideanDistance(X[i] , mus[0]); 
		d_mu2 = euclideanDistance(X[i] , mus[1]);
		if(d_mu1 < d_mu2){
			closerToMu1[i] = true;	
		} else { // in this case d_mu1 >= d_mu2
			closerToMu1[i] = false;
		} 
	}
}
	
int chooseBestSplit(vector< vector<double> > Xs, vector<int> clean_split_candidates){
    cout << "chooseBestSplit: number of candidates: "<<clean_split_candidates.size()<<endl;
    /* project X's onto splitting dimension */
	vector<double> projectedXs;
	int npts = Xs.size();
	//cout <<"chooseBestSplit: number of input points = "<<npts<<endl;
	double sumsqdists, minsumsqdists=numeric_limits<double>::max();
	int bestSplitDim = -1;

	/* assumes clean_split_candidates are features for which
        the data has at least 2 values */
	for(int j=0; j<clean_split_candidates.size(); j++){
		int splitting_dim = clean_split_candidates[j];
        for(int i=0; i<npts; i++){
			projectedXs.push_back(Xs[i][splitting_dim]);
		}
        
		/* split points by 2-means in one dimension */
        struct timeval twoMeansStart, twoMeansFinish;
        gettimeofday(&twoMeansStart, NULL);
		pair< double, double > sqdistsmidptpair = twoMeansOneD(projectedXs);//twomeans(X);
        gettimeofday(&twoMeansFinish, NULL);
        double twoMeansTime = twoMeansFinish.tv_sec - twoMeansStart.tv_sec;
        //cout << " two means time = " << twoMeansTime << endl;
        sumsqdists = sqdistsmidptpair.first;
		//cout << "chooseBestSplit: sumsqdists at dimension"
		//<<splitting_dim<<" = "<<sumsqdists<<endl;
		if(sumsqdists < minsumsqdists){
			minsumsqdists = sumsqdists;
			bestSplitDim = splitting_dim;
		}
		projectedXs.clear();
	}
	//cout << "chooseBestSplit: best splitting dim = "<<bestSplitDim;
	//cout <<", minsumsqdists = "<<minsumsqdists<<endl;
	return bestSplitDim;
}

vector<double> projectOntoOneDim(vector< vector<double> > X, int splitdim){
	if(splitdim<0){
		//cout << "splitting dimension must be >=0 "<<endl;
		exit(0);
	}
	vector<double> projectedXs;
	for(int i=0; i<X.size(); i++){
		projectedXs.push_back(X[i][splitdim]);
	}
	return projectedXs;
}

/* TODO: projectOntoLinearComboOfDims */

TwoMeansTreeNode * buildTwoMeansTree(vector<int> &indices, vector< vector<double> > &X, unsigned int d, unsigned int depth_threshold, int idparent, int m_try){
	int npts = X.size();
	//cout << "X.size() = "<<X.size()<<endl;
	//cout << "indices.size() = "<<indices.size()<<endl;
	int min_pts_in_leaf = 1;
	//cout << "npts = "<<X.size()<<endl;
	
    /* split criteria: stop splitting when number
	 *	of points in a node is low, or when
	 * 	depth limit is met
	*/
    struct timeval createLeafNodeStart, createLeafNodeFinish;
    gettimeofday(&createLeafNodeStart, NULL);
    if(d>=depth_threshold || npts <= min_pts_in_leaf){
		//cout << "d>="<<depth_threshold<<" or "<<npts<<"<="<<min_pts_in_leaf<<endl;
		//cout << "creating new leaf node with "<<npts<<" points "<<endl;
		TwoMeansTreeNode * leafnode = new TwoMeansTreeNode(X,indices, d, true, idparent);
        gettimeofday(&createLeafNodeFinish, NULL);
        double createLeafNodeTime = createLeafNodeFinish.tv_sec - createLeafNodeStart.tv_sec;
        cout << "create leaf node time = " << createLeafNodeTime <<endl;
		return leafnode;
	}
	
	int ndimensions;
	if(npts>0){
		ndimensions = (X[0]).size(); //assumes X's are all same dimensionality
    }
	
	/* Choose a random (with replacement) subset of fixed size
	   representing splitting dimension candidates */
	vector<int> splitting_dim_candidates;
	vector<int> dimensions;
    vector<double> projectedXs;
	for(int j=0; j<ndimensions; j++){
		dimensions.push_back(j);	
	}
    
    /* If the points all have the same value in a dimension,
     then discard this dimension from the set of candidates */
    for(int j=0; j<dimensions.size(); j++){
        int splitting_dim = dimensions[j];
        for(int i=0; i<npts; i++){
            projectedXs.push_back(X[i][splitting_dim]);
        }
        
        /* Only add dimension as a split candidate
            if sample instances take on at least two distinct
            values in this dimension. */
        bool allsame = true;
        auto it = projectedXs.begin();
        double sameval = *(projectedXs.begin());
        while(allsame == true && it != projectedXs.end()){
             ++it;
             double val = *it;
             if(val !=  sameval){
                allsame = false;
                //cout << "val = "<<val<<"; sameval = "<<sameval<<endl;
             }
        }
        if(!allsame){
         splitting_dim_candidates.push_back(splitting_dim);
        }
        projectedXs.clear();
    }
    cout << "number of splitting dimension candidates with at least two values: " << splitting_dim_candidates.size()<<endl;
    
    if(splitting_dim_candidates.size()==0){
        cout << "all dimensions have the same value for all instances; creating leaf node" <<endl;
        TwoMeansTreeNode * leafnode = new TwoMeansTreeNode(X, indices, d, true, idparent);
        vector<int> uniqueindices;
        uniqueindices = indices;
        sort(uniqueindices.begin(), uniqueindices.end());
        auto last = unique(uniqueindices.begin(), uniqueindices.end());
        uniqueindices.erase(last,uniqueindices.end());
        return leafnode;
    }
    
	/* shuffle the dimensions to get a random sample */
    struct timeval randomShuffleStart, randomShuffleFinish;
    gettimeofday(&randomShuffleStart, NULL);
	random_shuffle(splitting_dim_candidates.begin(), splitting_dim_candidates.end());
    gettimeofday(&randomShuffleFinish, NULL);
    double randomShuffleTime = randomShuffleFinish.tv_sec - randomShuffleFinish.tv_sec;
    //cout << "random shuffle time = "<<randomShuffleTime<<endl;
    
	/* mtry is the number of dimensions to test
	*	with one-dimensional k-means
	*/
    int mtry = m_try;
	//cout << "mtry = "<<subset_dims_size<<endl;
    vector<int> subset_split_dim_candidates;
    for(int m=0; m<mtry; m++){
        subset_split_dim_candidates.push_back(splitting_dim_candidates[m]);
    }
    
    /* test each splitting dimension from set of candidates,
     * choose the best splitting dimension in terms of k-means
     *  optimality */
    struct timeval chooseSplitStart, chooseSplitFinish;
    gettimeofday(&chooseSplitStart, NULL);
    int splitting_dim = chooseBestSplit(X, subset_split_dim_candidates);
    gettimeofday(&chooseSplitFinish, NULL);
    double chooseSplitTime = chooseSplitFinish.tv_sec - chooseSplitStart.tv_sec;
    cout << "choose best split time = "<< chooseSplitTime << endl;
    //cout << "splitting dimension at depth "<<d<<" = "<<splitting_dim<<endl;
	
	/* project onto splitting dimension */	
	//cout << " projecting data onto splitting dimension "<<endl;
	projectedXs = projectOntoOneDim(X, splitting_dim);
	
	/* perform 2-means clustering of data in one dimension */
	//cout << "computing means with 2-means"<<endl;
	pair< double, double> meanspair = twoMeansOneD(projectedXs);//twomeans(X);
	double midpt = meanspair.second;
	vector< vector<double> > midptVec;
	vector<double> midptV;
	midptV.push_back(midpt);
	midptVec.push_back(midptV);
	
	/* split data based on 2-means partitioning */
	//cout << "splitting data"<<endl;
	bool closertoMu1[npts];
    struct timeval splitData2MeansStart, splitData2MeansFinish;
    gettimeofday(&splitData2MeansStart, NULL);
	split_1D_DataBy2Means(X, midpt, closertoMu1, splitting_dim);
    gettimeofday(&splitData2MeansFinish, NULL);
    double splitData2MeansTime = splitData2MeansFinish.tv_sec - splitData2MeansStart.tv_sec;
    //cout << "split data by 2-means time = "<<splitData2MeansTime<<endl;
    
	//cout << "split data of size "<<npts<<"by 2-means."<<endl;	
	vector< vector<double> > leftsplit;
	vector<int> leftindices;
	vector< vector<double> > rightsplit;
	vector<int> rightindices;
	int nleft=0, nright=0;
	for(int i=0; i<X.size(); i++){
		if(closertoMu1[i]){
			leftsplit.push_back(X[i]);
			leftindices.push_back(indices[i]);
			nleft++;
		} else {
			rightsplit.push_back(X[i]);
			rightindices.push_back(indices[i]);
			nright++;
		}
	}
	if(leftsplit.size()==0 || rightsplit.size()== 0){ // in this case all 1-D projections were equivalent
		TwoMeansTreeNode * leafnode = new TwoMeansTreeNode(X, indices, d, true, idparent);
		return leafnode;
	}
	//cout << "splitting: "<<nleft<<" points left and "<<nright<<" points right"<<endl;
	
	/* recurse on left and right sides of tree */
	TwoMeansTreeNode * leftsubtree = buildTwoMeansTree(leftindices, leftsplit, d+1, depth_threshold, idparent+1, m_try);
	TwoMeansTreeNode * rightsubtree = buildTwoMeansTree(rightindices, rightsplit, d+1, depth_threshold, idparent+2, m_try);
	TwoMeansTreeNode * root = new TwoMeansTreeNode(midptVec, indices, d, false, idparent);
	root->setLeftChild(leftsubtree);
	//cout << "set left child "<<endl;
	root->setRightChild(rightsubtree);
	//cout << "set right child"<<endl;
	root->setSplitDim(splitting_dim);
	return root;	
}

/* numPoints: returns the number of points
	stored in the leaves of a TwoMeansTreeNode 
	tree
*/
int numPoints(TwoMeansTreeNode* tree){
	if(tree->getLeftChild() == NULL && tree->getRightChild()==NULL){
		vector< vector<double> > leafpoints = tree->getPoints();
		return leafpoints.size();
	} else {
		return (numPoints(tree->getLeftChild()) + numPoints(tree->getRightChild()));
	}
}

/* getRandomSample:
 *	return a sample of a given size, drawn from X
 *	with replacement
 */
vector< vector<double> > getRandomSample(vector< vector<double> > X, int size, int ** appearsInTree, int treeID){
	vector< vector<double> > Xs;
	vector< vector<double> > subsetXs;
	for(int i=0; i<X.size(); i++){
		Xs.push_back(X[i]);
	}
	for(int i=0; i<size; i++){
		int randidx = rand()%size;
		/*cout << "adding random point (";
		for(int j=0; j<Xs[randidx].size(); j++){
			cout << Xs[randidx][j]<<" ";
		}
		cout <<")"<<endl;*/
		subsetXs.push_back(Xs[randidx]);
		appearsInTree[randidx][treeID]++;
	}
	return subsetXs;
}

/* getRandomSampleIndices: 
*	return a vector of integers that is a
* 	random sample, with replacement, from 
*	the numbers 0 to n
*/
vector<int> getRandomSampleIndices(int n, int ** appearsInTree, int treeID){
	vector<int> indices;
	for(int i=0; i<n; i++){
		int randidx = rand()%n;
		//cout << "  adding random point "<< randidx << endl; 
		indices.push_back(randidx);
		appearsInTree[randidx][treeID]++;
	}
	return indices;
}

vector< TwoMeansTreeNode * > buildRandomForest(vector< vector<double> > X, int numTrees, unsigned int depthThreshold, int ** appearsInTree){
    int ndimensions = (X[0]).size();
    int mtry = (int) sqrt(ndimensions) ;
    bool fixed_mtry = false;
    int fixedmtryval = 4;
    if(fixed_mtry){
        mtry = fixedmtryval;
        cout << "fixing mtry to " <<mtry<<endl;
    }
    cout << "mtry = "<<mtry<<endl;
	vector< TwoMeansTreeNode* > forest;
	for(int i=0; i<numTrees; i++){
		/* bagging: get a random sample, with replacement, from X */
		vector<int> randindices = getRandomSampleIndices(X.size(), appearsInTree, i);
		cout << "sample size for tree "<<i<<" = "<<randindices.size()<<endl;
		vector< vector<double> > sampleXs;
		for(int j=0; j<randindices.size(); j++){
			sampleXs.push_back( X[randindices[j]] );
		}
        struct timeval buildTreeStart, buildTreeFinish;
        gettimeofday(&buildTreeStart, NULL);
        TwoMeansTreeNode * tree = buildTwoMeansTree(randindices, sampleXs, 0, depthThreshold, 0, mtry);
		gettimeofday(&buildTreeFinish, NULL);
        double buildTreeTime = buildTreeFinish.tv_sec - buildTreeStart.tv_sec;// + (buildTreeFinish.tv_usec - buildTreeStart.tv_usec)/1000000;
        forest.push_back(tree);
		cout << "finished tree "<<i<<endl;
        cout << "time to build tree "<<i<<": "<< buildTreeTime << " seconds "<< endl;
	}
	return forest;
}

TwoMeansTreeNode* classLeaf(vector<double> x, TwoMeansTreeNode* t){
	if(t->isLeafNode()){	
		return t;	
	} else {
		int splitdim = t->getSplitDim();
		double midpt = t->getMidpoint();
		if(x[splitdim] < midpt){
			return classLeaf(x, t->getLeftChild());
		} else {
			return classLeaf(x, t->getRightChild());
		}	
	}
}

/* kNNEstSim: returns the indices of the k nearest neighbors
        in the data set, based on their estimated similarity
        to the point indexed by i, as determined by the
        unsupervised random forest */
vector< int > kNNEstSim(int i, int k, int datasetsize){
    vector< pair<int, double> > neighbors;
    double estimated_sim_ij;
    
    for(int j=0; j<datasetsize; j++){
        pair<int, int> ijpair;
        if(j<i){
            ijpair = make_pair(i,j);
        } else if (i<j) {
            ijpair = make_pair(j,i);
        }
        auto found = coOccurMap.find(ijpair);
        if ( found == coOccurMap.end() ){
            estimated_sim_ij = 0;
            cout << "pair ("<<i<<","<<j<<") not found"<<endl;
        } else {
            estimated_sim_ij = coOccurMap[ijpair];
        }
        //cout << "estimated similarity to point "<<j<<": "<<estimated_sim_ij<<endl;
        pair<int, double> j_pair = make_pair( j, estimated_sim_ij );
        neighbors.push_back(j_pair);
    }
    sort(neighbors.begin(), neighbors.end(), greaterSecondInPair);
    /*for(int j = 0; j<10; j++){
        cout << "near neighbor "<<j<<":"<<(neighbors[j]).first<<"; similarity: "<<(neighbors[j]).second<<endl;
    }*/
    vector<int> knn;
    for(int j = 0; knn.size()<k; j++){
        knn.push_back((neighbors[j]).first);
    }
    return knn;
}

vector< int > kNNEuclidean(int i, int k, vector< vector<double> > & data){
    vector< pair<int, double> > neighbors;
    for(int j=0; j<data.size(); j++){
        pair<int, double> jpair = make_pair(j, euclideanDistance(data[i], data[j]) );
        neighbors.push_back(jpair);
    }
    sort(neighbors.begin(), neighbors.end(), secondInPair);
    vector<int> knn;
    for(int j=0; j<neighbors.size(); j++){
        knn.push_back((neighbors[j]).first);
    }
    return knn;
}

vector< vector<double> > kNearestNeighbors(vector<double> x, int k, vector<TwoMeansTreeNode*> forest){
	vector< vector<double> > neighbors;
	const int ntrees = forest.size();
	/* run the point down each tree, and record how many
		times each point co-occurs with the point 
		the nearest neighbors are those which 
		co-occured most frequently with the input
		point x */
	vector< vector<double> > leafpoints;
	vector< pair<vector<double>, int> > coOccurCounts; 
	vector< vector<double> > uniqueNeighbors;
	int maxCoOccurrences = 0;
	
	
	for(int t=0; t<ntrees; t++){
		TwoMeansTreeNode* leaf = classLeaf(x, forest[t]);
		vector< vector<double> > leafpoints = leaf->getPoints();
		for(int i=0; i<leafpoints.size(); i++){
			vector<double> y = leafpoints[i];
  			vector< vector<double> >::iterator found = find(uniqueNeighbors.begin(), uniqueNeighbors.end(), y);
			//if ( found == coOccurMap.end() ){
			if ( found == uniqueNeighbors.end() && y != x ){
				//coOccurMap.insert(make_pair< vector<double>, int>(y, 1));
				uniqueNeighbors.push_back(y);
				const vector<double> y2 = y;
				pair< vector<double> , int > newpair = make_pair(y2, 1);
				coOccurCounts.push_back(newpair);
				if(maxCoOccurrences==0){
					maxCoOccurrences=1;
				}
			} else if( y != x) { /* ensure that nearest neighbor isn't the point itself */
				//coOccurMap.at(y)++;
				int foundidx = found - uniqueNeighbors.begin();
				(coOccurCounts[foundidx]).second++;
				if(coOccurCounts[foundidx].second > maxCoOccurrences){
					//maxCoOccurrences = coOccurMap.at(y);
					maxCoOccurrences = (coOccurCounts[foundidx]).second;
				}
			}
		}
	}
	
	//for(auto& y::coOccurMap){
	//for(int i=0; i<coOccurCounts.size(); i++){
		//pair< vector<double>, int> y = coOccurCounts[i];
	
	sort(coOccurCounts.begin(), coOccurCounts.end(), occurCountGreaterThan);
	
	for(int j=0; j < k; j++){
		neighbors.push_back((coOccurCounts[j]).first);
	}
	
	return neighbors;	
}

vector<double> nearestNeighbor(vector<double> x, vector<TwoMeansTreeNode*> forest){
	vector< vector<double> > neighbors;
	const int ntrees = forest.size();
	/* run the point down each tree, and record how many
		times each point co-occurs with the point 
		the nearest neighbors are those which 
		co-occured most frequently with the input
		point x */
	vector< vector<double> > leafpoints;
	//unordered_map< vector<double> , int> coOccurMap;
	vector< pair<vector<double>, int> > coOccurCounts;
	//vector< pair<vector<double>, int> > coOccurCounts(sizey, make_pair(vector<int>(y), 0)); 
	vector< vector<double> > uniqueNeighbors;
	int maxCoOccurrences = 0;
	for(int t=0; t<ntrees; t++){
		TwoMeansTreeNode* leaf = classLeaf(x, forest[t]);
		vector< vector<double> > leafpoints = leaf->getPoints();
		for(int i=0; i<leafpoints.size(); i++){
			vector<double> y = leafpoints[i];
			//unordered_map< vector<double>, int >::const_iterator found = coOccurMap.find(y);
  			vector< vector<double> >::iterator found = find(uniqueNeighbors.begin(), uniqueNeighbors.end(), y);
			//if ( found == coOccurMap.end() ){
			if ( found == uniqueNeighbors.end() && y != x ){
				//coOccurMap.insert(make_pair< vector<double>, int>(y, 1));
				uniqueNeighbors.push_back(y);
				const vector<double> y2 = y;
				pair< vector<double> , int > newpair = make_pair(y2, 1);
				coOccurCounts.push_back(newpair);
				if(maxCoOccurrences==0){
					maxCoOccurrences=1;
				}
			} else if( y != x) { /* ensure that nearest neighbor isn't the point itself */
				//coOccurMap.at(y)++;
				int foundidx = found-uniqueNeighbors.begin();
				(coOccurCounts[foundidx]).second++;
				if(coOccurCounts[foundidx].second > maxCoOccurrences){
					//maxCoOccurrences = coOccurMap.at(y);
					maxCoOccurrences = (coOccurCounts[foundidx]).second;
				}
			}
		}
	}
	
	//for(auto& y::coOccurMap){
	for(int i=0; i<coOccurCounts.size(); i++){
		pair< vector<double>, int> y = coOccurCounts[i];
		if(y.second == maxCoOccurrences){
			neighbors.push_back(y.first);
		}
	}
	
	if(neighbors.size()>1){
		random_shuffle(neighbors.begin(), neighbors.end());
	}
	vector<double> nearestNeighbor = neighbors[0];
	return nearestNeighbor;
}
	
bool appearInSameLeafNode(vector<double> a, vector<double> b, TwoMeansTreeNode* tree){
	bool foundadim[a.size()];
	bool foundbdim[b.size()];
	bool fa, fb, founda, foundb, foundaAndb=false;
	if( tree->isLeafNode() ){
		vector< vector<double> > pointsInLeafNode = tree->getPoints();
		founda = (find(pointsInLeafNode.begin(), pointsInLeafNode.end(), a)!=pointsInLeafNode.end());
		foundb = (find(pointsInLeafNode.begin(), pointsInLeafNode.end(), b)!=pointsInLeafNode.end());
		return ( founda && foundb );
	} else {
		int dim = tree->getSplitDim();
		double midpt = tree->getMidpoint();
		if( (a[dim] < midpt && b[dim] >= midpt) || (a[dim] >= midpt && b[dim] < midpt)){
			return false; // points are not on same side of split
		} else {
			return (appearInSameLeafNode( a,b,tree->getLeftChild() ) || appearInSameLeafNode( a,b,tree->getRightChild() ) );
		}
	}
}

/* code from : https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c */
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
vector< vector<double> > read_mnist(string folder)
{
    vector< vector<double> > Xs;
    string fname = folder+"/t10k-images-idx3-ubyte";
    cout << "filename = "<<fname<<endl;
    ifstream file (fname);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            vector<double> Xvec;
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    cout << "char = "<<temp<<endl;
		    cout << " (double) char = "<<(double) temp <<endl;
		    Xvec.push_back((double) temp);
		}
            }
	    //cout << "Xvec.size() = "<<Xvec.size()<<endl;
	    Xs.push_back(Xvec);
        }
    file.close();
    } else {
	cout << "failed to open file"<<endl;	
    }
    cout << "read MNIST data"<<endl;
    return Xs;
}

void printEstimatedSimilarities(string ofss_string, int datasetsize, int **appearsInTree, int ntrees, vector< TwoMeansTreeNode * > RF, vector< vector<double> > X){
    if(datasetsize<1){
        cout << "printEstimatedSimilarities: Error: dataset size < 1"<<endl;
        exit(0);
    }
    
    /* for each pair of points, print out the number of times the two points
     appeared in the same leaf node divided by the number of times
     the two points were both used to build a tree */
    double estimated_sim_ij=0.0;
    ofstream est_sim_file;
    est_sim_file.open(ofss_string.c_str());
    for(int i=0; i<datasetsize; i++){
        //cout << "point "<<i<<": ";
        for(int j=0; j<datasetsize; j++){
            int treeappearances = 0;
            estimated_sim_ij = 0;
            for(int t=0; t<ntrees; t++){
                if(appearsInTree[i][t]>0 && appearsInTree[j][t]>0){
                    treeappearances++;
                }
            }
            /* initialize estimated similarity between points i and j
                to be the number of times i and j co-occur in the same
                leaf node */
            pair<int, int> ijpair;
            if(j<i){
                ijpair = make_pair(i,j);
            } else if (i<j) {
                ijpair = make_pair(j,i);
            }
            estimated_sim_ij = coOccurMap[ijpair];
            if(estimated_sim_ij - treeappearances > 0){//>treeappearances && treeappearances >0){
                cout << "Error: estimated similarity > tree co-appearances"<<endl;
                cout << "\ti = "<<i<<", j = "<<j;
                cout << ", tree appeareances = "<<treeappearances<<", ";
                cout << "co-occurrences = "<<estimated_sim_ij<<endl;
                exit(0);
            }
            /* estimated similarity is number of co-occurrences
                of i and j in the same leaf node divided by the
                number of times i and j occurr in the same tree */
            if(treeappearances>0){
                estimated_sim_ij /= (double) treeappearances;
            }
            if(i==j){
                estimated_sim_ij = 1;
            }
            coOccurMap[ijpair] = estimated_sim_ij;
            est_sim_file << estimated_sim_ij<<"\t";
        }
        est_sim_file<<endl;
    }
    est_sim_file.close();
    cout << "printEstimatedSimilarites: printed estimated similarities to file. "<<endl;
}

/*  eliminateSingleValuedDimensions: input: nxd data set X
		eliminates all dimensions i that only take a
		single value v(i) for all n instances.
		returns the vector that contains all dimensions
		for which instances take at least 2 values  */
vector< vector<double> > eliminateSingleValuedDimensions(vector< vector<double> > Y){
	vector< vector<double> > cleanY;
	int n = Y.size(); //dataset size
	
	if(n < 1){
		cout << "eliminateSingleValuedDimension: input data size is <1; returning empty vector.";
		return cleanY;
	}	
	
	int d = Y[0].size(); //d is dimensionality of data; assumes all instances have same dimensionality
	
	bool oneValue[d]; 
     	for(int j = 0; j < d; j++){
             	oneValue[j]=true;
     	}	

	// find dimensions where instances only take one value
	int countSingleValued=0;
	for(int j=0; j<d; j++){
		double val = Y[0][j];
		for(int i=1; i<n; i++){
			if(oneValue[j] &&  val!=Y[i][j]){
				oneValue[j] = false;
			}
		}
		if(oneValue[j]){ 
			cout << "dimension "<<j<<" has only a single value for all instances"<<endl;
			countSingleValued++;
		}
	}
	
	// create new, cleaned-up version of input with single-valued dimensions removed
	int newd = d - countSingleValued;
	for(int i=0; i<n; i++){
		vector<double> Y_i;
		for(int j=0; j<d; j++){
			if(!oneValue[j]){
				Y_i.push_back(Y[i][j]);
			}
		}
		assert(Y_i.size() == newd);
		cleanY.push_back(Y_i);
	} 
	assert(n == cleanY.size());
	
	return cleanY;
}

void printPointTreeAppearances(string ofname, int **appearsInTree, int datasetsize, int ntrees){

	ofstream appear_in_tree_file;
	appear_in_tree_file.open(ofname.c_str());
	for(int i=0; i<datasetsize; i++){
		for(int j=0; j<ntrees; j++){
			appear_in_tree_file << appearsInTree[i][j]<<"\t";
		}
		appear_in_tree_file <<endl;
	}
	appear_in_tree_file.close();	
	cout << "printed point-tree appearances"<<endl;
}

void updateSimMtx(TwoMeansTreeNode* t){
    
    if(t->isLeafNode()){ /* leaf case */
        vector<int> uniqueindices = t->getPointIndices();
        struct timeval sortUniqueIndicesStart, sortUniqueIndicesFinish;
        gettimeofday(&sortUniqueIndicesStart, NULL);
        sort(uniqueindices.begin(), uniqueindices.end());
        gettimeofday(&sortUniqueIndicesFinish, NULL);
        double sortUniqueIndicesTime = sortUniqueIndicesFinish.tv_sec - sortUniqueIndicesStart.tv_sec;
        //cout << "sort Unique Indices time = " << sortUniqueIndicesTime <<endl;
        auto last = unique(uniqueindices.begin(), uniqueindices.end());
        uniqueindices.erase(last,uniqueindices.end());
        
        for(int it = 0; it!=uniqueindices.size(); it++){
            int i = uniqueindices[it];
            for(int it2 = 0; it2<it; it2++){
                int j = uniqueindices[it2];
                int numcooccurrences=0;
                pair <int, int> ijpair;
                if(j<i){
                    ijpair = make_pair(i,j);
                } else {
                    ijpair = make_pair(j,i);
                }
                auto found = coOccurMap.find(ijpair);
                if ( found == coOccurMap.end() ){
                    coOccurMap[ijpair] = 1;
                } else if (i<j) {
                    coOccurMap[ijpair]++;
                }
            }
        }
    } else { /* non-leaf node case */
        updateSimMtx(t->getLeftChild());
        updateSimMtx(t->getRightChild());
    }
}

void simMtx(vector< TwoMeansTreeNode* > rf){
    /* for each tree, and each leaf node, update the co-occur map */
    for(int t=0; t<rf.size(); t++){
        TwoMeansTreeNode* tr = rf[t];
        updateSimMtx(tr);
    }
}

// test driver function
int main(int argc, char **argv){
	unsigned int treedepth;
	if(argc < 4 || argc > 6)
	{
		printf("Usage : ./testTwoMeansForest <tree depth> <data set size (number of points)> <number of trees> [(optional) input training file] [(optional) input test file] \n");
		exit(-1);
	}
	bool readInData = false;
    bool readTestData = false;
	if(argc == 5 || argc == 6){
		readInData = true;
    } else if(argc == 6) {
        readTestData = true;
    }
	
	string mnist_data_loc = "/Users/veronikastrnadova-neeley/Documents/U-ReRF/MNIST_data";
	//cout << "MNIST data directory: " << mnist_data_loc << endl;

	treedepth = (unsigned int)atoi(argv[1]);
	int datasetsize = atoi(argv[2]);
    cout << "tree depth: "<< treedepth<<"; data set size = "<<datasetsize<<" points. "<<endl;

	// small simple data test
	/*
	int Xarr[] = {1, 5, 7.8, 10.2, 15, 19, 21, 199, 200, 201, 202, 203, 204, 205}; 
	vector<double> X(Xarr, Xarr + sizeof(Xarr)/sizeof(Xarr[0]));
	vector<int> idxsX = (1,2,3,4,5,6,7,8,9,10,11,12,13,14);
	TwoMeansTreeNode* tree = buildTwoMeansTree(idxs, X, 0, 4, 0);
	cout << "Tree 1: "<< endl;
	printTree(tree);
	*/
	
    struct timeval readDataStart, readDataFinish;
    gettimeofday(&readDataStart, NULL);
	ifstream inFile;
	vector< vector<double> > Y;
	vector<int> indices;
	//int ndims = atoi(argv[1]);
    int ndims=-1;
	
	bool readMNISTData = false;
	
	if(readMNISTData){
		cout << "reading in MNIST data..."<<endl;
		Y = read_mnist(mnist_data_loc);
	} else if(readInData){
		string filename = argv[4];
		size_t found = filename.find_first_of(".");
		string extension = filename.substr(found, 4);
		cout << "extension = "<<extension<<endl;
		if( strcmp(extension.c_str(),".txt") == 0){
			inFile.open(filename.c_str());
			if (!inFile) {
        			cout << "Unable to open text file";
        			exit(1); // terminate with error
            }
			cout << "reading in data..."<<endl;
			double x;
			vector<double> temp;
            vector<string> line;
            string str;
            while(inFile){//} >> x){
                if(!getline(inFile, str)) break;
                boost::split(line, str, boost::is_any_of(" "));
                if(ndims == -1) ndims = line.size()-1;
                if(line.size() != ndims){
                    cout << "Error: Varying number of dimensions in input file."<<endl;
                    exit(0);
                }
                for(int idx=0; idx<ndims; idx++){ // ignore dimensions beyond the number of dimensions in the first line
                    x = atof((line[idx]).c_str());
                    temp.push_back(x);
                }
                if(ndims != -1 && temp.size()==ndims){
                    Y.push_back(temp);
                    temp.clear();
                }
			}
			for(int i=0; i<Y.size(); i++){
				indices.push_back(i);
			}
			inFile.close();
			cout << "read in text input data; size = "<<Y.size()<<" points."<<endl;
		} else if ( strcmp(extension.c_str(), ".csv" ) == 0) {
			inFile.open(filename.c_str());
            if (!inFile) {
                    cout << "Unable to open csv file"<<endl;
        			exit(1); // terminate with error
    			}
			cout << "reading in data..."<<endl;
			vector<string> line;
			string str;
			double x;
			vector<double> temp;
			while( inFile.good() ){
				//getline ( inFile, str, ',' ); // read a string until next comma
     				//cout << string( val, 1, val.length()-2 );
                if(!getline(inFile, str)) break;
                boost::split(line, str, boost::is_any_of(", "));
                if(ndims == -1) ndims = line.size()-1;
                if(line.size()-1 != ndims && ndims !=-1){
                    cout << "Error: varying number of dimensions."<<endl;
                    cout << "line size =" << line.size()-1<<endl;
                    cout << "ndims = "<<ndims<<endl;
                }
				for(int idx=0; idx<ndims; idx++){
					x = atof((line[idx]).c_str());
					temp.push_back(x);
					if(temp.size()==ndims){
						Y.push_back(temp);
						temp.clear();
					}
				}
			}
			for(int i=0; i<Y.size(); i++){
				indices.push_back(i);
			}
			inFile.close();
			cout << "read in csv input data; size = "<<Y.size()<<" points."<<endl;
            cout << "dimensionality of data = "<<ndims<<endl;
		} else if( strcmp(extension.c_str(),".bin") == 0 ) {
			cout << "reading in CIFAR binary file..."<<endl;
			ifstream inFile2;
			inFile2.open(filename.c_str(), ios::in | ios::binary | ios::ate);	
			if(!inFile2){
				cout << "Unable to open CIFAR binary file."<<endl;
				exit(0);
			}
			int fileSize = inFile2.tellg();
			cout << "file size = "<< fileSize <<" bytes "<<endl;
			//std::unique_ptr<char[]> buffer(new char[fileSize]);
    			char * buffer = new char[fileSize];
			inFile2.seekg(0, std::ios::beg);
    			inFile2.read(buffer, fileSize);
			inFile2.close();
			cout << "read in CIFAR binary input data."<<endl;
			int num_images = 10;
			vector<double> temp;
			for(int i=0; i<num_images; i++){ //loop through the images
				for(int j=1; j<3073; j++){ // j=0 is the label
					unsigned char pixelVal = (unsigned char) buffer[i+3073+j];
					temp.push_back(pixelVal);
					if(j<10){ 
					//	cout << pixelVal << endl;
					}
				}
				Y.push_back(temp);
				temp.clear();
			}	
			cout << "stored pixel values"<<endl;
			cout << "number of images = "<<Y.size()<<endl;
			if(Y.size()>0){
				cout << "dimensionality = "<<Y[0].size()<<endl;
			}
			//exit(0);
		} else {
			cout << "cannot recognize file format."<<endl;
			exit(0);
		}
	} else {
		cout << "generating random data"<<endl;
        ndims = 10; /* fix number of dimensions manually in the case of random data */
        cout << "number of dimensions = "<<ndims;
		for(int i=0; i<datasetsize; i++){
			vector<double> temp;
			for(int j=0; j<ndims; j++){
				temp.push_back((double) rand() / RAND_MAX);
			}
			Y.push_back(temp);
			indices.push_back(i);
			temp.clear();
		}
	}
    gettimeofday(&readDataFinish, NULL);
    double readDataTime = readDataFinish.tv_sec - readDataStart.tv_sec;
    cout << "read data time = "<<readDataTime<<endl;
	cout << "data set size = "<<Y.size()<<endl;
	
	/* pre-process data: eliminate any dimension/feature that takes only one value */
	Y = eliminateSingleValuedDimensions(Y);
		
	const int ntrees = atoi(argv[3]);  //set number of trees manually for now
	int **appearsInTree = new int *[Y.size()]; /* store whether or not a
					 	was included in the sample
						used to build each tree */
	for(int i=0; i<Y.size(); i++){
		appearsInTree[i] = new int[ntrees];
		for(int j=0; j<ntrees; j++){
			appearsInTree[i][j] = 0;
		}
	}
	
	int **leafIDs = new int *[Y.size()]; /* store the leaf ID of each
						point in each tree */
	for(int i=0; i<Y.size(); i++){
		leafIDs[i] = new int[ntrees];
		for(int j=0; j<ntrees; j++){
			leafIDs[i][j] = -1; // -1 indicates point doesn't occur in this tree
		}
	}
       
	struct timeval buildForestStart, buildForestFinish;
	gettimeofday(&buildForestStart, NULL);
	vector< TwoMeansTreeNode* > random2meansforest = buildRandomForest(Y,ntrees, treedepth, appearsInTree);
	gettimeofday(&buildForestFinish, NULL);  
	double buildForestTime = buildForestFinish.tv_sec - buildForestStart.tv_sec;
	cout << "forest building time = "<<buildForestTime<<endl;
	
	/* print out whether a point appears in a particular tree */
	stringstream intreess;
	if(readInData){
		intreess<<"point_tree_appearances_"<<datasetsize<<"_pts"
		<<ndims<<"dimensions_depth"<<treedepth<<argv[4];
	} else {
		intreess<<"point_tree_appearances_"<<datasetsize<<"_pts"
		<<ndims<<"dimensions_depth"<<treedepth<<".txt";
	}
	printPointTreeAppearances(intreess.str(), appearsInTree, Y.size(), ntrees);
		
    /* update co-occur map */
    struct timeval updateCoOccurMapStart, updateCoOccurMapFinish;
    gettimeofday(&updateCoOccurMapStart, NULL);
    simMtx(random2meansforest);
    gettimeofday(&updateCoOccurMapFinish, NULL);
    double coOccurMapUpdateTime = updateCoOccurMapFinish.tv_sec - updateCoOccurMapStart.tv_sec;
    cout << "co-occur map update time: "<<coOccurMapUpdateTime<<endl;
    
	/* print out each tree in the forest */
	/*for(int i=0; i<random2meansforest.size(); i++){
		TwoMeansTreeNode * tree = random2meansforest[i];
		printTree(tree);
	}*/
	
    // print out pairwise estimated similarities
	stringstream ofss;
	if(readInData){
		ofss<<"estimatedsim_"<<datasetsize<<"_pts"
		<<ndims<<"dimensions_depth"<<treedepth<<"_"<<ntrees<<"trees"<<argv[5];
	} else {
		ofss<<"estimatedsim_"<<datasetsize<<"_pts"
		<<ndims<<"dimensions_depth"<<treedepth<<"_"<<ntrees<<"_trees"<<".txt";
	}
	cout << "printing estimated similarities to file: "<<ofss.str()<<endl;
	//printCoOccurMap(datasetsize);
    
    printEstimatedSimilarities(ofss.str(), Y.size(), appearsInTree, ntrees, random2meansforest, Y);
    struct timeval printingFinished;
    gettimeofday(&printingFinished, NULL);
    
	/* test random point for nearest neighbors */
    double precision=0;
    int nTestPts = 20;
    int k=5;
    int atN=5;
	for(int i=0; i<nTestPts; i++){
		int rand_idx = rand()%Y.size();
		vector<double> randpt = Y[rand_idx];
        cout << "random point index = "<<rand_idx<<endl;
		/*cout << "random point: (";
		for(int d=0; d<randpt.size(); d++){
			cout <<randpt[d]<<",";
		}
		cout <<")"<<endl;*/
		vector< int > nearestnbrs = kNNEstSim(rand_idx, k, Y.size());
		cout << "k nearest neighbors based on estimated similarity : (";
		for(int j=0; j<k; j++){
			vector<double> neighbor = Y[nearestnbrs[j]];
			/*for(int d=0; d<neighbor.size(); d++){
                cout <<neighbor[d];
				if(d!=neighbor.size()-1) cout <<",";
			}*/
            pair<int, int> jpair;
            if(rand_idx <nearestnbrs[j]){
                jpair = make_pair(nearestnbrs[j],rand_idx);
            } else if(nearestnbrs[j] < rand_idx) {
                jpair = make_pair(rand_idx,nearestnbrs[j]);
            } else {
                jpair = make_pair(rand_idx, rand_idx);
            }
            double estimated_sim = coOccurMap[jpair];
            cout << "neighbor "<<j<<": "<<nearestnbrs[j]<<", ";
            cout << "Euclidean distance: "<<euclideanDistance(randpt, neighbor)<<"; ";
            cout << "Estimated similarity: "<< estimated_sim << "; ";
		}
        cout <<")"<<endl;
        
        vector<int> euclideanNearestNeighbors = kNNEuclidean(rand_idx, atN, Y);
        cout << "Euclidean distance: "<<endl;
        for(int j=0; j<atN; j++){
            vector< double > eucNbr = Y[euclideanNearestNeighbors[j]];
            cout << "neighbor "<<j<<": "<<euclideanNearestNeighbors[j]<<", distance: "<<euclideanDistance(randpt, eucNbr)<<endl;
        }
        
        sort(nearestnbrs.begin(), nearestnbrs.end());
        sort(euclideanNearestNeighbors.begin(), euclideanNearestNeighbors.end());
        vector<int> intersectSet(k);
        vector<int>::iterator it = set_intersection(nearestnbrs.begin(), nearestnbrs.end(), euclideanNearestNeighbors.begin(), euclideanNearestNeighbors.end(), intersectSet.begin());
        intersectSet.resize(it - intersectSet.begin());
        
        cout << "Intersect set: ";
        for(it = intersectSet.begin(); it!= intersectSet.end(); it++){
            cout << *it<<", ";
        }
        cout <<endl;
        
        double precisionAtN = ((double) intersectSet.size())/((double) atN);
        cout << "Precision at " << atN << ": " << precisionAtN << endl;
        precision+=precisionAtN;
	}
    precision /= atN;
    cout << "Average precision = " << precision <<endl;
    
	double buildForestAndPrintTime = printingFinished.tv_sec - buildForestStart.tv_sec;
	//cout << "forest building and printing time = "<<buildForestAndPrintTime<<endl;
    
    
    
	return 0;
}
