#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define min(a,b) \
    ({ __typeof__ (a) _a = (a); \
           __typeof__ (b) _b = (b); \
                _a < _b ? _a : _b; })
#define max(a,b) \
    ({ __typeof__ (a) _a = (a); \
           __typeof__ (b) _b = (b); \
                _a > _b ? _a : _b; })

void snf_ensure_nb_divides(int n1, int n2, int M[n1][n2], 
                       int L[n1][n1], int LI[n1][n1],
                       int R[n2][n2], int RI[n2][n2], int s);

void print_mat(int n1, int n2, int mat[n1][n2]){
    int i,j;
    printf("------------------------------------------------\n");
    for(i=0;i<n1;i++){
        for(j=0;j<n2;j++){
            printf("%5d",mat[i][j]);
        }
        printf("\n");
    }
}
// ============================================================================
//
// Math functions
//
// ============================================================================
int rem(int a, int b){
    int r;
    r = a%b;
    if(r>0 && b<0) return r+b;
    if(r<0 && b>0) return r+b;
    return r;
}
int quo(int a, int b){
    int r1,r2;
    r1 = rem(a,b);
    r2 = a%b;
    if(rem(r2-r1,b)!=0) printf("wrong!!!!!%5d,%5d,%5d,%5d\n",a,b,r1,r2);
    return a/b+(r2-r1)/b;
}
int gcd(int a, int b){
    int r0, r1, r2;
    r0 = max(abs(a),abs(b));
    r1 = min(abs(a),abs(b));
    if(r1==0){
        return r0;
    }
    else{
        r2=rem(r0,r1);
        while(r2>0){r0=r1; r1=r2; r2=rem(r0,r1);}
        return r1;
    }
}
int lcm(int a, int b){
    return abs(a*b)/gcd(a,b);
}

// ============================================================================
//
// Auxillary functions
//
// ============================================================================
int max_value(int n1, int n2, int mat[n1][n2]){
    int vv, i, j;
    vv=0;
    for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
        if(vv<abs(mat[i][j])) vv=abs(mat[i][j]);
    }}
    return vv;
}

// ============================================================================
//
// Elemenatry matrix transformation
//
// ============================================================================
void transform_interchange_row(int n1, int n2, int mat[n1][n2], int i, int j){
    int tmp, k;
    if(i!=j){
        for(k=0;k<n2;k++){tmp=mat[i][k]; mat[i][k]=mat[j][k]; mat[j][k]=tmp; }
    }
}
void transform_invert_row(int n1, int n2, int mat[n1][n2], int i){
    int k;
    for(k=0;k<n2;k++){ mat[i][k]=-mat[i][k]; }
}
void transform_add_row(int n1, int n2, int mat[n1][n2], int i, int j, int c){
    int k;
    for(k=0;k<n2;k++){ mat[j][k]+=c*mat[i][k]; }
}
void transform_interchange_col(int n1, int n2, int mat[n1][n2], int i, int j){
    int tmp,k;
    if(i!=j){
        for(k=0;k<n1;k++){tmp=mat[k][i]; mat[k][i]=mat[k][j]; mat[k][j]=tmp;}
    }
}
void transform_invert_col(int n1, int n2, int mat[n1][n2], int i){
    int k;
    for(k=0;k<n1;k++){ mat[k][i]=-mat[k][i]; }
}
void transform_add_col(int n1, int n2, int mat[n1][n2], int i, int j, int c){
    int k;
    for(k=0;k<n1;k++){ mat[k][j]+=c*mat[k][i]; }
}


//=============================================================================
//
// Row echelon form by gaussian 
//
//=============================================================================
int ref_move_least_to_start(int n1, int n2, int M[n1][n2], int L[n1][n1], int LI[n1][n1],int s, int sj){
    int sj_, num, i, i0;
    for(sj_=sj;sj_<n2;sj_++){
        num=M[s][sj_];
        i0 =s;
        for(i=s+1;i<n1;i++){
            if(M[i][sj_]!=0){
                if(num==0 || abs(M[i][sj_])<abs(num)){
                    num=M[i][sj_];
                    i0 = i;
                }
            }
        }
        if(num!=0) break;
    }
    if(num==0) return sj_;
    //
    if(i0!=s){
        transform_interchange_row(n1,n2,M, s,i0);
        transform_interchange_row(n1,n1,L, s,i0);
        transform_interchange_col(n1,n1,LI,s,i0);
    }
    if(M[s][sj_]<0){
        transform_invert_row(n1,n2,M, s);
        transform_invert_row(n1,n1,L, s);
        transform_invert_col(n1,n1,LI,s);
    }
    return sj_;
}

void ref_modify_edge(int n1, int n2, int M[n1][n2], int L[n1][n1], int LI[n1][n1], int s, int sj){
    int i,q;
    for(i=0;i<n1;i++){
        if(i==s) continue;
        if(M[i][sj]!=0){
            q=quo(M[i][sj],M[s][sj]);
            transform_add_row(n1,n2,M, s,i,-q);
            transform_add_row(n1,n1,L, s,i,-q);
            transform_add_col(n1,n1,LI,i,s, q);
        }
    }
}


void ref_null_edge(int n1, int n2, int M[n1][n2], int L[n1][n1], int LI[n1][n1], int s, int sj){
    char null;
    int i, sj_;
    while(1){
        null=1;
        for(i=s+1;i<n1;i++){
            if(M[i][sj]!=0){ null=0; break;}
        }
        if(null) break;
        //
        sj_ = ref_move_least_to_start(n1,n2,M,L,LI,s,sj);
        if(sj_!=sj) printf("Wrong !!!! sj_!=sj\n");
        ref_modify_edge(n1,n2,M,L,LI,s,sj);
    }
}

int row_echelon_form(int n1, int n2, int M[n1][n2], int L[n1][n1], int LI[n1][n1]){
    int i,j,s,sj,rank=0;
    for(i=0;i<n1;i++){
    for(j=0;j<n1;j++){
        L[i][j] = (i==j)? 1:0;
        LI[i][j]= (i==j)? 1:0;
    }}
    //
    sj=0;
    for(s=0;s<n1;s++){
        sj = ref_move_least_to_start(n1,n2,M,L,LI,s,sj);
        if(sj==n2){ rank=s; break;}
        else if (s==n1-1) {rank=n1;}
        ref_modify_edge(n1,n2,M,L,LI,s,sj);
        ref_null_edge(n1,n2,M,L,LI,s,sj);
    }
    //
    return rank;
}

int row_echelon_form_interface(int n1, int n2, int M[], int L[], int LI[]){
    int (*M_)[n2] = (int (*)[n2])&M[0], 
        (*L_)[n1] = (int (*)[n1])&L[0], 
        (*LI_)[n1]= (int (*)[n1])&LI[0];
    //
    return row_echelon_form(n1,n2,M_,L_,LI_);
}

// ============================================================================
//
// Smith normal form
//
// ============================================================================

// Moves least element in south-western block, that starts at position [s, s] into start position
void snf_move_least_to_start(int n1, int n2, int M[n1][n2], 
                         int L[n1][n1], int LI[n1][n1],
                         int R[n2][n2], int RI[n2][n2], int s){
    int num, i, j, i0, j0;

    num = abs(M[s][s]);
    i0  = s; 
    j0  = s;

    for(i=s;i<n1;i++){
    for(j=s;j<n2;j++){
        if( M[i][j]!=0 && (num==0 || abs(M[i][j])<num) ){
            i0 = i; j0=j;
            num= abs(M[i][j]);
        }
    }}

    //print_mat(n1,n1,L);
    //print_mat(n1,n1,LI);
    if(i0!=s){
        transform_interchange_row(n1,n2,M, i0,s);
        transform_interchange_row(n1,n1,L, i0,s);
        transform_interchange_col(n1,n1,LI,i0,s);
    }

    if(j0!=s){
        transform_interchange_col(n1,n2,M, j0,s);
        transform_interchange_col(n2,n2,R, j0,s);
        transform_interchange_row(n2,n2,RI,j0,s);
    }

    if(M[s][s]<0){
        transform_invert_row(n1,n2,M, s);
        transform_invert_row(n1,n1,L, s);
        transform_invert_col(n1,n1,LI,s);
    }
    //print_mat(n1,n1,L);
    //print_mat(n1,n1,LI);
    //printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
}


// Modifies edging by standard operations in order to make it zero, should be followed by null_edging(
void snf_modify_edging(int n1, int n2, int M[n1][n2],
                   int L[n1][n1], int LI[n1][n1],
                   int R[n2][n2], int RI[n2][n2], int s){
    int i,q;

    //print_mat(n1,n1,L);
    //print_mat(n1,n1,LI);
    for(i=s+1;i<n1;i++){
        if(M[i][s]!=0){
            q = quo(M[i][s],M[s][s]);
            transform_add_row(n1,n2,M, s,i,-q);
            transform_add_row(n1,n1,L, s,i,-q);
            transform_add_col(n1,n1,LI,i,s, q);
        }
    }

    for(i=s+1;i<n2;i++){
        if(M[s][i]!=0){
            q = quo(M[s][i],M[s][s]);
            transform_add_col(n1,n2,M, s,i,-q);
            transform_add_col(n2,n2,R, s,i,-q);
            transform_add_row(n2,n2,RI,i,s, q);
        }
    }
    //print_mat(n1,n1,L);
    //print_mat(n1,n1,LI);
    //printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
}

//Moves least element in edging, that starts at position [s, s] into start position
void snf_move_le_to_start(int n1, int n2, int M[n1][n2],
                      int L[n1][n1], int LI[n1][n1],
                      int R[n2][n2], int RI[n2][n2], int s){
    int num, i, j, i0, j0;
    
    num = abs(M[s][s]);
    i0  = s;
    j0  = s;

    for(i=s+1;i<n1;i++){
        if(M[i][s]!=0 && abs(M[i][s])<num){
            num = M[i][s];
            i0  = i;
            j0  = s;
    }}

    for(j=s+1;j<n2;j++){
        if(M[s][j]!=0 && abs(M[s][j])<num){
            num = abs(M[s][j]);
            i0  = s;
            j0  = j;
    }}
    
    if(j0==s && i0>s){
        transform_interchange_row(n1,n2,M, i0,s);
        transform_interchange_row(n1,n1,L, i0,s);
        transform_interchange_col(n1,n1,LI,i0,s);
    }else if(i0==s && j0>s){
        transform_interchange_col(n1,n2,M, j0,s);
        transform_interchange_col(n2,n2,R, j0,s);
        transform_interchange_row(n2,n2,RI,j0,s);
    }
}


// Makes edging, that starts at pos [s, s] zero, ensures south-western block does not divide element at pos [s, s]
void snf_null_edging(int n1, int n2, int M[n1][n2],
                 int L[n1][n1], int LI[n1][n1],
                 int R[n2][n2], int RI[n2][n2], int s){
    char null_row, null_col;
    int  i;

    while(1){
        null_col = 1;
        for(i=s+1;i<n1;i++){
            if(M[i][s]!=0){ null_col=0; break;}
        }
        null_row = 1;
        for(i=s+1;i<n2;i++){
            if(M[s][i]!=0){ null_row=0; break;}
        }
        if(null_row && null_col) break;
        //
        snf_move_le_to_start(n1,n2,M,L,LI,R,RI,s);
        snf_modify_edging(n1,n2,M,L,LI,R,RI,s);
    }
    //
    if(M[s][s]<0){
        transform_invert_row(n1,n2,M, s);
        transform_invert_row(n1,n1,L, s);
        transform_invert_col(n1,n1,LI,s);
    }
    //
    snf_ensure_nb_divides(n1,n2,M,L,LI,R,RI,s);
}

// Ensures south-western block does not divide element at pos [s, s]
void snf_ensure_nb_divides(int n1, int n2, int M[n1][n2], 
                       int L[n1][n1], int LI[n1][n1],
                       int R[n2][n2], int RI[n2][n2], int s){
    int num, i, j, i0, j0, num_, q;
    //
    num = M[s][s];
    i0  = s; 
    j0  = s;
    for(i=s+1;i<n1;i++){
    for(j=s+1;j<n2;j++){
        num_ = rem(M[i][j],M[s][s]);
        if(num_!=0 && num_<num){
            i0=i; j0=j;
            num = num_;
        }
    }}
    if(i0>s || j0>s){
        transform_add_row(n1,n2,M, i0,s,1 );
        transform_add_row(n1,n1,L, i0,s,1 );
        transform_add_col(n1,n1,LI,s,i0,-1);
        //
        q = quo(M[s][j0],M[s][s]);
        transform_add_col(n1,n2,M,s,j0,-q);
        transform_interchange_col(n1,n2,M,s,j0);
        transform_add_col(n2,n2,R,s,j0,-q);
        transform_interchange_col(n2,n2,R,s,j0);
        transform_add_row(n2,n2,RI,j0,s,q);
        transform_interchange_row(n2,n2,RI,j0,s);
        //
        snf_null_edging(n1,n2,M,L,LI,R,RI,s);
    }
}

// Iteration of transformation into Smith normal form, modifies edging starting at position [s, s]
void transform_smith(int n1, int n2, int M[n1][n2], 
                     int L[n1][n1], int LI[n1][n1],
                     int R[n2][n2], int RI[n2][n2], int s){
    snf_move_least_to_start(n1,n2,M,L,LI,R,RI,s);
    snf_modify_edging(n1,n2,M,L,LI,R,RI,s);
    snf_null_edging(n1,n2,M,L,LI,R,RI,s);
}

// Checks whether south-western block of Mix, starting at pos [s, s], is zero
char snf_block_empty_or_null(int n1, int n2, int M[n1][n2], int s){
    char null=1;
    int i,j;
    for(i=s;i<n1;i++){
        for(j=s;j<n2;j++){
            if(M[i][j]!=0) {null=0;break;}
        }
        if (!null) break;
    }
    return null;
}

// Computes Smith's normal form B of Mix A, such that L * A * R = B
// returns:
//  - Mix in Smith normal form (B)
//  - L square Mix (L)
//  - R square Mix (R)
//  - rank of Mix
int smith_form(int n1, int n2, int M[n1][n2], 
                       int L[n1][n1], int LI[n1][n1],
                       int R[n2][n2], int RI[n2][n2]){
    int s,rank,i,j;
    //
    for(i=0;i<max(n1,n2);i++){
    for(j=0;j<max(n1,n2);j++){
        if(i<n1 && j<n1){
            L[i][j] = i==j? 1:0;
            LI[i][j]= i==j? 1:0;
        }
        if(i<n2 && j<n2){
            R[i][j] = i==j? 1:0;
            RI[i][j]= i==j? 1:0;
        }
    }}
    //
    rank=0;
    for(s=0;s<min(n1,n2);s++){
        if(snf_block_empty_or_null(n1,n2,M,s)) break;
        transform_smith(n1,n2,M,L,LI,R,RI,s);
        rank+=1;
        /*if(n1==88 && n2==46){
            printf("-------------------------------------------------------------------\n");
            printf("%5d,  %10d,   %10d,  %10d\n", s, M[s][s], max_value(n1,n1,L), max_value(n2,n2,R));
        }*/
    }
    return rank;
}


int smith_form_interface(int n1, int n2, int M[], 
                         int L[], int LI[], int R[], int RI[]){
    int (*M_)[n2] = (int (*)[n2])&M[0], 
        (*L_)[n1] = (int (*)[n1])&L[0], 
        (*LI_)[n1]= (int (*)[n1])&LI[0], 
        (*R_)[n2] = (int (*)[n2])&R[0],
        (*RI_)[n2]= (int (*)[n2])&RI[0];
    //
    return smith_form(n1,n2,M_,L_,LI_,R_,RI_);
}
