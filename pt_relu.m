function out = pt_relu(vector)
%Function returns the point-wise arctan of the input gector
	len = length(vector);
	out = zeros(1,len);
	for i=1:len
		if(vector(i)>0)
            out(i)=vector(i);
        end
	end
end