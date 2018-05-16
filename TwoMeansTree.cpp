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

using namespace std;

class TwoMeansTreeNode {
	private:
		TwoMeansTreeNode *left;
		TwoMeansTreeNode *right;
		unsigned int depth;
		double midpoint;
		//vector< double > means;
		//double means[2];
		vector< vector<double> > points;
		int ndimensions;
		bool leafNode;
	public: 
		TwoMeansTreeNode(vector< vector<double> > pts, unsigned int d, bool isLeafNode){
			if(isLeafNode){
			    //set points at this node
			    for(int i=0; i<pts.size();i++){
			    	points.push_back(pts[i]);
			    } 
			    leafNode = true;
			    //cout << "added points to leaf node"<<endl;
			} else {
				leafNode = false;
				assert(pts.size() == 1);
				if((pts[0]).size() != 1){
					cout << "(pts[0]).size() = "<<(pts[0]).size()<<endl;
					cout << "Internal node only stores one point, the midpoint. "<<endl;
				}
				midpoint = pts[0][0];
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
		
};

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
		cout << "Leaf node "//<< tree->getID() 
			<<"points at depth "<<tree->getDepth() <<": <";
		for(int i=0; i<pts.size(); i++){
			cout << "(";
			for(int j=0; j<(pts[i]).size(); j++){
				cout << ", " << (pts[i])[j];
			}
			cout <<"), ";
		}
		cout <<">"<<endl;
	} else {
		cout <<"Internal node "//<< tree->getID() 
			<<" at depth " << tree->getDepth() 
			<<" midpoint: "<<tree->getMidpoint()<<endl;
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
 *	means, and the means themselves
 */
pair< double, double > twoMeansOneD(vector<double> X){
	int npts = X.size();
 	/* process points in sorted order */	
	sort(X.begin(), X.end());
	bool isCloserToMean1[npts];
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
	int seenpts=0;
	double sumsqdists=0.0, minsumsqdists=numeric_limits<double>::max();
	while(seenpts<(npts-1)){
		it++;
		seenpts++;
		/* adjust sums */
		sum1+=*it;
		sum2-=*it;
		/* adjust means */
		mean1 = sum1/(double) seenpts;
		mean2 = sum2/(double) (npts-seenpts);
		cout << "iteration "<<seenpts<<": means found: mean1="<<mean1<<", mean2="<<mean2<<endl;
		cout << " sum1="<<sum1<<", sum2="<<sum2<<endl;	
		
	
		/* assign points to closest mean and update sumsqdists */
		sumsqdists=0.0;
		for(int i=0; i<npts; i++){
			double dist1 = fabs(X[i] - mean1);
			double dist2 = fabs(X[i] - mean2);
			if(dist1<dist2){
				sumsqdists += fabs(X[i]-mean1)*fabs(X[i]-mean1);
				isCloserToMean1[i] = true;
			} else {
			   	sumsqdists += fabs(X[i]-mean2)*fabs(X[i]-mean2);
				isCloserToMean1[i] = false;	
			}
		}
	
		/* store means if sum of squared distances
		* to means yields best split
		*/
		if(sumsqdists<minsumsqdists){
			minsumsqdists = sumsqdists;
			means.clear();
			means.push_back(mean1);
			means.push_back(mean2);
			midpoint = (mean1+mean2)/2;
			cout << "min sum sq. dists = "<<minsumsqdists<<endl;
			//cout << ", mean1 = "<<mean1<<", mean2 = "<<mean2<<endl;
		}
	}	
	if(means.size()>1)
	cout << "twoMeansOneD: returning midpont " <<midpoint<<endl;
	return make_pair(minsumsqdists, midpoint);
}

vector< vector<double> > twomeans(vector< vector<double> > X){	
	int npts = X.size();
	int ndims=0;
	if(!X.empty()){
		ndims = (X[0]).size();
	} 
	bool isCloserToMean1[npts];	
	//choose starting means at random from the points in the data set
	int idx1 = rand() % npts;
	int idx2 = rand() % npts;	
	vector<double> mean1 = X[idx1];
	vector<double> mean2 = X[idx2];
	//keep track of how many swaps happened 
	// between this iteration and the last
	int nSwaps = npts;
	int iters=0, maxiterations=1000;
	while (nSwaps>0 || iters>=maxiterations){ //iterate until no points are re-assigned
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
	
int chooseBestSplit(vector< vector<double> > Xs, vector<int> splitting_dim_candidates){
	// project X's onto splitting dimension
	vector<double> projectedXs;
	int npts = Xs.size();
	double sumsqdists, minsumsqdists=numeric_limits<double>::max();
	int bestSplitDim = -1;
	/* test each splitting dimension from set of candidates,
	* save the best splitting dimension in terms of k-means
	*  optimality */
	for(int j=0; j<splitting_dim_candidates.size(); j++){
		int splitting_dim = splitting_dim_candidates[j];
		for(int i=0; i<npts; i++){
			projectedXs.push_back(Xs[i][splitting_dim]);
			
			/* split points by 2-means in one dimension */	
			pair< double, double > sqdistsmidptpair = twoMeansOneD(projectedXs);//twomeans(X);
			sumsqdists = sqdistsmidptpair.first;
			cout << "chooseBestSplit: sumsqdists at dimension"
				<<splitting_dim<<" = "<<sumsqdists<<endl;
			if(sumsqdists < minsumsqdists){
				minsumsqdists = sumsqdists;
				bestSplitDim = splitting_dim;
			}
		}
		projectedXs.clear();
	}
	cout << "chooseBestSplit: best splitting dim = "<<bestSplitDim;
	cout <<", minsumsqdists = "<<minsumsqdists<<endl;
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

TwoMeansTreeNode * buildTwoMeansTree(vector< vector<double> > X, unsigned int d, unsigned int depth_threshold){
	int npts = X.size();
	int min_pts_in_leaf = 1;
	//cout << "npts = "<<X.size()<<endl;
	/* split criteria: stop splitting when number 
	 *	of points in a node is low, or when
	 * 	depth limit is met
	*/
	if(d>=depth_threshold || npts <= min_pts_in_leaf){
		//cout << "d>="<<depth_threshold<<" or "<<npts<<"<=2"<<endl;
	}
	if(d>=depth_threshold || npts <= min_pts_in_leaf){
		//cout << "creating new leaf node"<<endl;
		TwoMeansTreeNode * leafnode = new TwoMeansTreeNode(X, d, true);
		return leafnode;
	}
	
	int ndimensions;
	if(npts>0){
		ndimensions = X[0].size(); //assumes X's are all same dimensionality
	}
	
	/* Choose a random (with replacement) subset of fixed size
	   representing splitting dimension candidates */
	vector<int> splitting_dim_candidates;
	vector<int> dimensions;
	for(int j=0; j<ndimensions; j++){
		dimensions.push_back(j);	
	}
	
	/* shuffle the dimensions to get a random sample */
	random_shuffle(dimensions.begin(), dimensions.end());
	
	/* subset_dims_size is the number of dimensions to test 
	*	with one-dimensional k-means
	*/
	int subset_dims_size = (int) sqrt(ndimensions);
	//cout << "Dimension subset size = "<<subset_dims_size<<endl;
	for(int j=0; j<subset_dims_size; j++){
		//cout << "adding dimension "<<dimensions[j]<<endl;
		splitting_dim_candidates.push_back(dimensions[j]);
	}
	//cout << endl;
	
	/* choose best splitting dimension from among candidates */
	int splitting_dim = chooseBestSplit(X, splitting_dim_candidates);
	cout << "splitting dimension at depth "<<d<<" = "<<splitting_dim<<endl;
	
	/* project onto splitting dimension */	
	//cout << " projecting data onto splitting dimension "<<endl;
	vector<double> projectedXs = projectOntoOneDim(X, splitting_dim);
	
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
	split_1D_DataBy2Means(X, midpt, closertoMu1, splitting_dim);
	//splitDataBy2Means(X, means, closertoMu1);
	//cout << "split data of size "<<npts<<"by 2-means."<<endl;	
	vector< vector<double> > leftsplit;
	vector< vector<double> > rightsplit;
	int nleft=0, nright=0;
	for(int i=0; i<X.size(); i++){
		if(closertoMu1[i]){
			leftsplit.push_back(X[i]);
			nleft++;
		} else {
			rightsplit.push_back(X[i]);
			nright++;
		}
	}
	//cout << "splitting: "<<nleft<<" points left and "<<nright<<" points right"<<endl;
	
	/* recurse on left and right sides of tree */
	TwoMeansTreeNode * leftsubtree = buildTwoMeansTree(leftsplit, d+1, depth_threshold);
	TwoMeansTreeNode * rightsubtree = buildTwoMeansTree(rightsplit, d+1, depth_threshold);
	TwoMeansTreeNode * root = new TwoMeansTreeNode(midptVec, d, false);
	root->setLeftChild(leftsubtree);
	//cout << "set left child "<<endl;
	root->setRightChild(rightsubtree);
	//cout << "set right child"<<endl;
	
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
vector< vector<double> > getRandomSample(vector< vector<double> > X, int size){
	vector< vector<double> > Xs;
	vector< vector<double> > subsetXs;
	for(int i=0; i<X.size(); i++){
		Xs.push_back(X[i]);
	}
	for(int i=0; i<size; i++){
		int randidx = rand()%size;
		cout << "adding random point (";
		for(int j=0; j<Xs[randidx].size(); j++){
			cout << Xs[randidx][j]<<" ";
		}
		cout <<")"<<endl;
		subsetXs.push_back(Xs[randidx]);
	}
	return subsetXs;
}

vector< TwoMeansTreeNode * > buildRandomForest(vector< vector<double> > X, int numTrees, unsigned int depthThreshold){
	
	vector< TwoMeansTreeNode* > forest;
	for(int i=0; i<numTrees; i++){
		/* bagging: get a random sample, with replacement, from X */
		vector< vector<double> > sampleXs = getRandomSample(X, X.size());
		TwoMeansTreeNode * tree = buildTwoMeansTree(sampleXs, 0, depthThreshold);
		forest.push_back(tree);
		cout << "finished tree "<<i<<endl;
	}
	return forest;
}
	
bool appearInSameLeafNode(vector<double> a, vector<double> b, TwoMeansTreeNode* tree){

	bool foundadim[a.size()];
	bool foundbdim[b.size()];
	bool fa, fb, founda, foundb, foundaAndb=false;
	if(tree->getLeftChild() == NULL && tree->getRightChild()==NULL){
		/*cout << " a\t\tb "<<endl;
		for(int i=0; i<a.size(); i++){
			cout <<a[i]<<"\t\t"<<b[i]<<endl;
		}*/
		//cout <<endl;
		vector< vector<double> > pointsInLeafNode = tree->getPoints();
		//cout << "leaf points: "<<endl;
		for(int i=0; i<pointsInLeafNode.size(); i++){
			//cout <<" point "<<i<<": (";
			for(int j=0; j<a.size(); j++){
				foundadim[j] = false;
				foundbdim[j] = false;
			}
			fa=true; 
			fb=true;
			for(int j=0; j<(pointsInLeafNode[i]).size(); j++){
				//cout <<(pointsInLeafNode[i])[j]<<", ";
				if( (pointsInLeafNode[i])[j] == a[j] ){
					foundadim[j] = true;	
				} else {
					foundadim[j] = false;
				}
				if( (pointsInLeafNode[i])[j] == b[j] ){
					foundbdim[j] = true;
				} else {
					foundbdim[j] = false;
				}
			}
			//cout <<"), ";
			 
			for(int j=0; j<a.size(); j++){
				if(!foundadim[j]) fa=false;
				if(!foundbdim[j]) fb=false;
			}
			if(fa && fb){
				foundaAndb=true;
			} else if(fa){
				//cout << "found a"<<endl;
				founda=true;
			} else if(fb){
				foundb=true;
				//cout << "found b"<<endl;
			}
		}
		//cout <<endl;
		/*if(founda){
			cout<<"found point a"<<endl;
		}
		if(foundb){
			cout<<"found point b"<<endl;
		}
		if( founda && foundb ) cout << "found points a and b"<<endl;
		*/
		return ( founda && foundb );
	} else {
		return (appearInSameLeafNode( a,b,tree->getLeftChild() ) || appearInSameLeafNode( a,b,tree->getRightChild() ) );
	}
}

// test driver function
int main(int argc, char **argv){
	unsigned int treedepth;
	
	if(argc < 4 || argc>5)
	{
		printf("Usage : ./testTwoMeansForest <number of dimensions> <tree depth> <data set size (number of points)> [(optional) inputfile]\n");
		exit(-1);
	}
	bool readInData = false;
	ifstream inFile;
	if(argc == 5){
		readInData = true;
	}	

	treedepth = (unsigned int)atoi(argv[2]);
	int datasetsize = atoi(argv[3]);

	// small simple data test
	/*
	int Xarr[] = {1, 5, 7.8, 10.2, 15, 19, 21, 199, 200, 201, 202, 203, 204, 205}; 
	vector<double> X(Xarr, Xarr + sizeof(Xarr)/sizeof(Xarr[0]));
	TwoMeansTreeNode* tree = buildTwoMeansTree(X, 0, 4);
	cout << "Tree 1: "<< endl;
	printTree(tree);
	*/
		
	vector< vector<double> > Y;
	int ndims = atoi(argv[1]);
	if(readInData){
		inFile.open(argv[4]);
		if (!inFile) {
        		cout << "Unable to open file";
        		exit(1); // terminate with error
    		}
		cout << "reading in data..."<<endl;
		double x;
		vector<double> temp;
		while(inFile >> x){	
			temp.push_back(x);
			if(temp.size()==ndims){
				Y.push_back(temp);
				temp.clear();
			}
		}
		inFile.close();
		cout << "read in input data"<<endl;
	} else {
		cout << "generating random data"<<endl;
		for(int i=0; i<datasetsize; i++){
		vector<double> temp;
			for(int j=0; j<ndims; j++){
				temp.push_back((double) rand() / RAND_MAX);
			}
			Y.push_back(temp);
			temp.clear();
		}
	}
	//cout << " generating tree2..."<<endl;
	TwoMeansTreeNode * tree2 = buildTwoMeansTree(Y, 0, treedepth);
	cout << "Tree 2: "<< endl;
	//printLevelOrder(tree2);
        printTree(tree2);	
	//cout << "Tree 2: number of points in tree = "<<numPoints(tree2)<<endl;

	for(int i=0; i<Y.size(); i++){
		cout << "point "<<i<<": (";
		for(int j=0; j<Y[i].size(); j++){
			cout << Y[i][j]<<", ";
		}
		cout <<")"<<endl;
	}
		
	int ntrees = 100;
	vector< TwoMeansTreeNode* > random2meansforest = buildRandomForest(Y,ntrees, treedepth);

	/* print out each tree in the forest */
	for(int i=0; i<random2meansforest.size(); i++){
		TwoMeansTreeNode * tree = random2meansforest[i];
		printTree(tree);
	}
		
	// print out pairwise estimated similarities as well as true distances
	double estimated_sim_ij=0.0;
	stringstream ofss;
	if(readInData){
		ofss<<"estimatedsim_"<<datasetsize<<"_pts"
		<<ndims<<"dimensions_depth"<<treedepth<<argv[4];
	} else {
		ofss<<"estimatedsim_"<<datasetsize<<"_pts"
		<<ndims<<"dimensions_depth"<<treedepth<<".txt";
	}
	ofstream est_sim_file;
	est_sim_file.open(ofss.str().c_str());
	double true_dist_ij;
	for(int i=0; i<datasetsize; i++){
		for(int j=0; j<datasetsize; j++){
			estimated_sim_ij=0;
			for(int k=0; k<ntrees; k++){	
				if(appearInSameLeafNode(Y[i],Y[j],random2meansforest[k])){
					estimated_sim_ij++;
					/*cout << "found "<<Y[i]<<" and "<<Y[j]
						<< " in same node in tree"
						<<k<<endl;*/
				}
			}
			estimated_sim_ij /= (double) ntrees;
			est_sim_file << estimated_sim_ij <<endl;
		}
		/*cout <<"finished printing similarities and true distances for point "<<i<<": (";
		for(int d=0; d<ndims; d++){
			cout<<Y[i][d]<<",";
		}
		cout <<")";*/
	}
	est_sim_file.close();
	
	return 0;
}
